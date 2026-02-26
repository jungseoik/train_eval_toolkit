## 기존 라벨링을 jsonl로 변환코드

python main.py label2jsonl -i "data/processed/gangnam/yeoksam2_v2/Test/video/falldown" -o "data/instruction/evaluation/test_gangnam_yeoksam2_v2_video_falldown.jsonl" -dt "video" -opt "test" -ity "clip" -itk "caption" -tn "falldown" 
python main.py label2jsonl -i "data/processed/gangnam/yeoksam2_v2/Test/video/violence" -o "data/instruction/evaluation/test_gangnam_yeoksam2_v2_video_violence.jsonl" -dt "video" -opt "test" -ity "clip" -itk "caption" -tn "violence" 

### 현대백화점 6번쨰 01.27 이미지 추가
# data/processed/hyundai_backhwajum/hyundai_01_27_QA/test
python main.py label2jsonl -i "data/processed/hyundai_backhwajum/hyundai_01_27_QA/train" -o "data/instruction/train/train_hyundai_01_27_QA_image_falldown.jsonl" -dt "image" -opt "train" -ity "capture_frame" -itk "caption" -tn "falldown"
python main.py label2jsonl -i "data/processed/hyundai_backhwajum/hyundai_01_27_QA/test" -o "data/instruction/evaluation/test_hyundai_01_27_QA_image_falldown.jsonl" -dt "image" -opt "test" -ity "capture_frame" -itk "caption" -tn "falldown"
## 테스트셋 학습 구축
python main.py label2jsonl -i "data/processed/hyundai_backhwajum/hyundai_01_27_QA/test" -o "data/instruction/train/test_hyundai_01_27_QA_image_falldown.jsonl" -dt "image" -opt "train" -ity "capture_frame" -itk "caption" -tn "falldown"
