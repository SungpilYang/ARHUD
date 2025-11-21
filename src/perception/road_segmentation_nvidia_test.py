import torch
import numpy as np
import cv2
import os
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from google.colab.patches import cv2_imshow
from PIL import Image

# ------------------------------------------------------------
# 1️. 상수 설정
# ------------------------------------------------------------
ROAD_INDEX = 0  # Cityscapes 기준 도로 인덱스는 0 입니다.
SEGFORMER_MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"


# ------------------------------------------------------------
# 2️. 모델 불러오기
# ------------------------------------------------------------
def load_model():
    """Hugging Face Hub에서 SegFormer 모델을 로드합니다."""
    print(f"Loading SegFormer Model: {SEGFORMER_MODEL_NAME}...")
    
    #  수정됨: SegFormer 전용 클래스 사용 
    processor = SegformerFeatureExtractor.from_pretrained(SEGFORMER_MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL_NAME)
    
    model.eval()
    print("> SegFormer Model loaded successfully.")
    return processor, model

# ------------------------------------------------------------
# 3️. 도로 세그멘테이션 및 원본 크기 복원
# ------------------------------------------------------------
def segment_road(processor, model, image_bgr):
    """
    SegFormer를 사용하여 추론하고 결과를 원본 이미지 크기로 복원합니다.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # GPU 설정 및 데이터 이동
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # warn 전처리: SegFormer Feature Extractor 사용
    # cv2 이미지를 PIL Image로 변환하여 입력 (FeatureExtractor의 요구사항)
    image_pil = Image.fromarray(image_rgb)
    inputs = processor(images=image_pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 추론
    with torch.no_grad():
        outputs = model(**inputs)
        
    # 로짓(logits)을 원본 이미지 크기로 업샘플링 (복원)
    logits = outputs.logits
    upsampled = torch.nn.functional.interpolate(
        logits, size=image_rgb.shape[:2], mode="bilinear", align_corners=False
    )
    # 최종 예측 마스크 (클래스 인덱스)
    pred = upsampled.argmax(dim=1)[0].cpu().numpy()

    # Cityscapes 기준 road = 0
    road_mask_resized = (pred == ROAD_INDEX).astype(np.uint8) * 255
    
    # 디버깅 강화: 예측된 클래스 확인
    unique_classes, counts = np.unique(pred, return_counts=True)
    print(f" 예측된 클래스 (인덱스: 픽셀 수): {list(zip(unique_classes, counts))}")

    # 오버레이 적용
    overlay = image_bgr.copy()
    overlay[road_mask_resized > 0] = (0, 255, 0) 
    blended = cv2.addWeighted(image_bgr, 0.6, overlay, 0.4, 0)
    
    return road_mask_resized, blended

# ------------------------------------------------------------
# 4️. 테스트 실행 (경로 설정은 동일)
# ------------------------------------------------------------
def test_run():
    base_dir = "./ARHUD/data"
    img_path = os.path.join(base_dir, "road_test.png")
    save_path = os.path.join(base_dir, "road_segmented_segformer.png")

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"warn 이미지가 없습니다! 경로 확인: {img_path}")

    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
         raise ValueError(f"warn 이미지 파일이 손상되었거나 로딩 실패: {img_path}")

    print(f"✅ Image loaded: {image_bgr.shape}")

    processor, model = load_model()
    mask, blended = segment_road(processor, model, image_bgr)

    cv2.imwrite(save_path, blended)
    print(f"> Saved to: {save_path}")
    
    cv2_imshow(blended)
    print("> 도로 세그멘테이션 완료!")


if __name__ == "__main__":
    test_run()