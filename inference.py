import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time

# -------------------------------------------
# 1) FUNÇÃO PARA CLASSIFICAR (shelf_model)
# -------------------------------------------
def classify_image(shelf_model, image, device):
    """
    Classifica a imagem usando o modelo shelf_model.
    
    Retorna:
        label (str): 'normal', 'suspect' ou 'unknown'
        confidence (float): Confiança da classificação
    """
    results = shelf_model.predict(image, device=device, verbose=False)
    if results and len(results):
        result = results[0]
        class_id = result.probs.top1
        confidence = result.probs.top1conf.item()  # Converte tensor para float

        if int(class_id) == 0:
            label = 'normal'
        else:
            # Ajuste de threshold de confiança, se necessário
            if confidence >= 0.6:
                label = 'suspect'
            else:
                label = 'normal'
        return label, confidence
    return 'unknown', 0.0

# -------------------------------------------
# 2) FUNÇÃO IOU PARA RASTREAMENTO
# -------------------------------------------
def iou(boxA, boxB):
    """
    Calcula o Intersection over Union (IOU) entre duas bounding boxes (x1, y1, x2, y2).
    """
    # Coordenadas da interseção
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    if (boxAArea + boxBArea - interArea) == 0:
        return 0.0
    return interArea / float(boxAArea + boxBArea - interArea)

# -------------------------------------------
# 3) FUNÇÃO PRINCIPAL
# -------------------------------------------
def main():
    print("Iniciando script...")

    # Seleciona o dispositivo (CUDA, MPS ou CPU)
    if torch.cuda.is_available():
        device = 'cuda'
        print("Usando CUDA para computação.")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("Usando MPS para computação.")
    else:
        device = 'cpu'
        print("Usando CPU para computação.")

    # Carrega os modelos (ajuste os caminhos se necessário)
    yolo_model = YOLO('models/yolov8n.pt')
    shelf_model = YOLO('models/best.pt')
    print("Modelos carregados.")

    # Abre o stream RTSP
    rtsp_url = 'rtsp://admin:Pi3,1415@172.16.0.66:554/cam/realmonitor?channel=9&subtype=0'
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"Erro ao abrir o stream RTSP: {rtsp_url}")
        return

    # Obtém o FPS do vídeo e configura "pulo" de frames
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        print("Não foi possível recuperar o FPS do stream. Usando 30 FPS como padrão.")
        original_fps = 30
    target_fps = 9
    frame_skip_interval = max(int(original_fps // target_fps) - 1, 0)

    print(f"FPS Original: {original_fps}")
    print(f"FPS Alvo: {target_fps}")
    print(f"Intervalo de pulo de frames: {frame_skip_interval}")

    # Cria janela de exibição
    cv2.namedWindow('Detecção em Tempo Real', cv2.WINDOW_NORMAL)
    max_width = 2560
    max_height = 1440

    # Lista de rastreamento de suspeitos
    # Cada elemento é um dict com:
    #   'bbox': (x1, y1, x2, y2)
    #   'last_time_seen': tempo (time.time()) da última vez que apareceu
    #   'confidence': última confiança obtida
    #   'suspect_count': nº de frames consecutivos como "suspect"
    #   'suspect_end_time': até que momento (time.time()) é considerado suspeito
    suspect_tracks = []

    # Parâmetros
    iou_threshold = 0.5
    required_suspect_frames = 5  # só vira 'suspect' depois de X frames consecutivos
    suspect_hold_time = 3.0      # mantém a bbox vermelha por 3 segundos após confirmar "suspect"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fim do stream ou erro na leitura do frame.")
            break

        process_start_time = time.time()

        # Executa detecção com YOLO (classes de pessoas)
        results = yolo_model.predict(frame, conf=0.3, device=device, verbose=False, imgsz=480)

        # Lista para as bounding boxes do frame atual
        current_bboxes = []

        # Para cada resultado de inferência no frame
        for result in results:
            for box in result.boxes:
                # Ignora qualquer classe que não seja 'pessoa' (classe 0 no COCO)
                if int(box.cls) != 0:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Limita coordenadas dentro do frame
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])

                if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                    continue

                # Classifica o recorte usando shelf_model
                cropped_img = frame[y1:y2, x1:x2]
                cropped_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                label, confidence = classify_image(shelf_model, cropped_rgb, device)

                # --------------------------------
                #   RASTREAMENTO E MARCAÇÃO
                # --------------------------------
                # Tenta achar um track com IoU acima de iou_threshold
                best_iou = 0.0
                matched_track = None
                for track in suspect_tracks:
                    iou_val = iou((x1, y1, x2, y2), track['bbox'])
                    if iou_val > iou_threshold and iou_val > best_iou:
                        best_iou = iou_val
                        matched_track = track

                if matched_track:
                    # Atualiza bounding box e tempo
                    matched_track['bbox'] = (x1, y1, x2, y2)
                    matched_track['last_time_seen'] = process_start_time

                    # Se a classificação atual for "suspect"
                    if label == 'suspect':
                        matched_track['suspect_count'] += 1

                        # Se atingiu frames consecutivos suficientes, ativa/renova o timer
                        if matched_track['suspect_count'] >= required_suspect_frames:
                            matched_track['suspect_end_time'] = process_start_time + suspect_hold_time
                    else:
                        # Se não for suspect neste frame, zera contagem consecutiva
                        matched_track['suspect_count'] = 0

                    # Verifica se ainda está no período "suspect"
                    if process_start_time < matched_track['suspect_end_time']:
                        current_label = 'suspect'
                        current_confidence = confidence
                    else:
                        current_label = 'normal'
                        current_confidence = 0.0

                else:
                    # Se não encontrou um track compatível
                    if label == 'suspect':
                        # Cria track novo
                        new_track = {
                            'bbox': (x1, y1, x2, y2),
                            'last_time_seen': process_start_time,
                            'confidence': confidence,
                            'suspect_count': 1,
                            'suspect_end_time': 0.0  # ainda não foi "confirmado"
                        }
                        suspect_tracks.append(new_track)
                        # Por enquanto, ainda não atingiu required_suspect_frames
                        current_label = 'normal'
                        current_confidence = 0.0
                    else:
                        # Se é normal, nem criamos track (ou crie se quiser rastrear tudo)
                        current_label = 'normal'
                        current_confidence = 0.0

                # Armazena pra desenhar depois
                current_bboxes.append(((x1, y1, x2, y2), current_label, current_confidence))

        # Limpeza de tracks antigos:
        # remove aqueles que não foram atualizados há mais de 0.5s (ajuste se quiser)
        new_tracks = []
        for track in suspect_tracks:
            if (process_start_time - track['last_time_seen']) <= 0.5:
                new_tracks.append(track)
        suspect_tracks = new_tracks

        # --------------
        # Desenho e Exibição
        # --------------
        original_height, original_width = frame.shape[:2]
        scale = min(max_width / original_width, max_height / original_height, 1)
        if scale < 1:
            display_frame = cv2.resize(
                frame, 
                (int(original_width*scale), int(original_height*scale)),
                interpolation=cv2.INTER_AREA
            )
        else:
            display_frame = frame.copy()

        for (bbox, label, confidence) in current_bboxes:
            x1, y1, x2, y2 = bbox
            x1_disp = int(x1 * scale)
            y1_disp = int(y1 * scale)
            x2_disp = int(x2 * scale)
            y2_disp = int(y2 * scale)

            if label == 'suspect':
                color = (0, 0, 255)  # vermelho
                thickness = 7
                cv2.rectangle(display_frame, (x1_disp, y1_disp), (x2_disp, y2_disp), color, thickness)
                cv2.putText(display_frame, f"suspect",
                            (x1_disp, y1_disp - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            else:
                color = (0, 255, 0)  # verde
                thickness = 2
                cv2.rectangle(display_frame, (x1_disp, y1_disp), (x2_disp, y2_disp), color, thickness)
                cv2.putText(display_frame, "normal", 
                            (x1_disp, y1_disp - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        process_end_time = time.time()
        fps = 1.0 / (process_end_time - process_start_time) if (process_end_time - process_start_time) > 0 else 0
        cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Mostra na janela
        cv2.imshow('Detecção em Tempo Real', display_frame)

        # Pula frames para atingir FPS alvo
        for _ in range(frame_skip_interval):
            ret = cap.grab()
            if not ret:
                break

        # Se apertar 'q', interrompe
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Finaliza
    cap.release()
    cv2.destroyAllWindows()
    print("Script finalizado.")

# -------------------------------------------
# 4) EXECUÇÃO DIRETA
# -------------------------------------------
if __name__ == '__main__':
    main()
