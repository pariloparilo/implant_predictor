# ResNet50 기준모델 실행 안내

> 2026-09-14: 이 문서는 **단일 데이터 루트의 검토된 4분할**에 대한 기존 학습 안내입니다. 제공 `1.Training/2.Validation` 자료의 첫 방문에는 [FIRST_VISIT_QUICKSTART.md](FIRST_VISIT_QUICKSTART.md)를 사용하세요. 제공 분할 기준 확인과 학습 연결 수정이 먼저 필요하며, 전체 자료를 바로 합쳐 재분할하지 않습니다.

현재 구현은 **검토한 전체 제품을 분류하는 closed-set 기준모델**입니다. 확률이 높다는 이유만으로 제품을 확정하거나 미학습 제품을 자동 판별하지 않습니다. calibration 집합은 추후 확률 보정·거절 기준 개발을 위해 보존합니다.

## 1. 현장 확인과 분할 확정

`implant_analysis/configs/site.example.json`을 `site.json`으로 복사합니다. 데이터 루트, 매칭 규칙, 기관에서 확인한 그룹키 CSV를 설정합니다. 먼저 부분 감사로 실제 JSON 키와 영상 대응을 확인하고, 확인 근거를 기록한 뒤 전체 감사와 분할 초안을 만듭니다. 경로·그룹키를 추측해서 verified를 켜지 않습니다.

아래 경로의 `/safe/work/...`는 예시입니다. 실제 실행 결과 경로로 바꿉니다. 상대경로는 **설정 파일이 있는 폴더 기준**입니다. work_dir는 원본 데이터 루트 밖이어야 합니다.

```bash
python -m implant_analysis audit --config implant_analysis/configs/site.json --run-id sample --limit 100
python -m implant_analysis audit --config implant_analysis/configs/site.json --run-id full
python -m implant_analysis split-draft --config implant_analysis/configs/site.json --audit-dir /safe/work/audit/full --run-id draft1
```

학습 전 `class_split_counts.csv`와 분할 요약을 검토합니다. 모든 제품이 train/dev에 있어야 현재 기준모델을 학습할 수 있습니다. 희귀 제품을 임의로 제거하거나 성능을 본 뒤 분할을 다시 고르지 않습니다. 전체 제품 기준 결과와 주평가 제품 범위의 구분은 팀이 학습 전에 정해야 하며, 현재 프로그램은 제품을 자동으로 주평가 범위로 제한하지 않습니다.

```bash
python -m implant_analysis freeze-split --config implant_analysis/configs/site.json --split-dir /safe/work/split_draft/draft1 --run-id study1 --review-note '실제 검토자·일자·환자/원촬영 단위·제품 범위·확인 근거를 여기에 기록'
```

출력 `frozen/study1`은 분할·제품 표와 검토 기록의 해시를 보관합니다. 파일이 변경되면 학습을 차단합니다. 이 기록은 연구 재현성을 위한 것이며 기관의 공식 승인을 대신하지 않습니다. 원촬영 단위 그룹만 있다면 환자 독립 성능이라고 표현하지 않습니다.

## 2. 학습 설정

`training.weights_path`에 승인받은 `resnet50-11ad3fa6.pth`의 로컬 경로, `weights_sha256`에 실제 파일의 전체 64자리 SHA-256을 넣습니다. 파일명 접미사 8자리는 전체 해시가 아닙니다. 해시 검증은 파일 동일성 검사이며 출처·반입 승인 확인도 별도로 필요합니다.

- ResNet50의 ImageNet 1,000개 클래스 가중치를 엄격히 읽은 뒤 출력층만 대상 제품 수로 교체합니다.
- 영상은 RGB로 변환하고 비율을 유지해 224×224에 검은 여백을 추가합니다. 임플란트 끝부분을 잘라내는 crop은 사용하지 않습니다.
- ImageNet 평균·표준편차로 정규화합니다. 학습에만 작은 회전(±3도)과 밝기/대비 변화(0.1)를 적용합니다. 좌우 반전·강한 왜곡·자동 품질 라벨은 적용하지 않습니다.
- 초기 2 epoch은 출력층을 학습하고 이후 layer4와 출력층을 학습합니다. 앞단과 모든 BatchNorm 통계는 고정합니다. 이는 계산량을 줄인 전이학습 기준안이며 전체 미세조정은 별도 실험입니다.
- AdamW, 출력층 학습률 1e-3/layer4 1e-4, 배치 16, 최대 10 epoch, dev macro-F1이 5 epoch 연속 개선되지 않으면 종료합니다. 클래스 가중치는 적용하지 않습니다.
- CUDA에서만 선택적으로 혼합 정밀도를 사용합니다. GPU 메모리가 부족하면 **새 실행 전에** 배치를 줄여 기록합니다. GPU가 없으면 `device`를 `cpu`로 명시합니다.

전처리는 torchvision의 일반 사진용 중심 crop과 다릅니다. 임플란트 전체 형상을 보존하기 위한 연구 설계 선택이며 우수성이 검증됐다는 뜻은 아닙니다. 모델 구조·가중치 API는 [torchvision ResNet 소스](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py), 안전한 state_dict·체크포인트 로딩은 [PyTorch 저장·재개 안내](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)를 참고했습니다.

## 3. 학습과 재개

```bash
python -m implant_analysis train --config implant_analysis/configs/site.json --frozen-dir /safe/work/frozen/study1 --run-id r50_run1
```

학습은 train/dev 영상만 읽습니다. calibration/test 영상은 무결성 검사에서도 열지 않습니다. 이미지가 감사 이후 바뀌면 중단합니다. 출력에는 설정, 환경 버전, 학습 기록, `last.pt`가 저장됩니다. `last.pt`에는 마지막 epoch의 모델·optimizer·scaler와 dev 기준 최적 모델이 함께 들어 있습니다. 평가 명령은 내부의 **최적 모델**을 사용합니다.

```bash
python -m implant_analysis train --config implant_analysis/configs/site.json --frozen-dir /safe/work/frozen/study1 --run-id r50_resume1 --resume /safe/work/train/r50_run1/last.pt
```

재개는 새 출력 폴더로 진행하며, 마지막 **완료 epoch**부터 시작합니다. epoch 도중 중단되면 해당 epoch을 다시 실행합니다. 같은 실행환경에서는 epoch별 seed로 자료 순서·증강을 재현하며, 재개와 연속 실행 일치를 합성 시험으로 점검합니다. 장비·CUDA 버전이 바뀐 경우 완전한 수치 일치를 보장하지 않습니다. 재개할 때 epochs 외 설정을 바꾸거나 조기 종료된 실험을 계속 이어갈 수 없습니다. 가중치와 체크포인트는 신뢰·승인된 파일만 사용하고 `weights_only=True`를 해제하지 않습니다.

`--synthetic-smoke`는 **합성 데이터로 코드 실행을 시험할 때만** 무작위 초기화를 허용하는 옵션입니다. 실제 분석에서는 사용하지 않습니다. 해당 표시는 체크포인트와 평가 결과에 남습니다.

## 4. 평가와 시각화

먼저 dev 결과로 코드와 산출물 형식을 확인합니다.

```bash
python -m implant_analysis evaluate --config implant_analysis/configs/site.json --frozen-dir /safe/work/frozen/study1 --checkpoint /safe/work/train/r50_run1/last.pt --partition dev --run-id r50_dev
python -m implant_analysis plot --evaluation-dir /safe/work/evaluate/r50_dev --training-dir /safe/work/train/r50_run1
```

출력은 다음과 같습니다.

- `metrics.json`: 정확도, Top-min(3,K) 정확도, 고정된 전체 제품 기준 macro-F1, 관측된 제품 기준 macro-F1/balanced accuracy, 제품별 지표, 혼동행렬.
- `predictions.csv`: 사례별 예측 확률과 그룹·병원·촬영방법. **식별 가능 정보를 포함할 수 있는 내부 전용 파일**입니다.
- `subgroups.json`: 병원/촬영방법별 같은 지표, 표본 수, 독립 그룹 수, 빠진 제품 목록. 작은 하위집단의 단순 비교를 일반화하지 않습니다.
- `figures/`: 정규화 혼동행렬과 학습 곡선(PNG/PDF). 한글 폰트가 없어도 재현되도록 클래스 번호를 쓰며 `class_map.json`에 대응표를 제공합니다.

모든 학습·분석 선택을 확정한 뒤에만 test를 평가합니다. 아래 메모는 단순 통과 문구가 아니라 실제 확정 근거를 기록하는 곳입니다.

```bash
python -m implant_analysis evaluate --config implant_analysis/configs/site.json --frozen-dir /safe/work/frozen/study1 --checkpoint /safe/work/train/r50_run1/last.pt --partition test --run-id r50_final --final-test-note '모델·전처리·평가 기준 확정 일자와 근거'
```

test를 평가한 뒤 모델·분할·임계값을 바꾸어 다시 같은 test에서 성능을 고르는 것은 허용되는 검증 설계가 아닙니다. 프로그램의 기록 장치만으로 팀의 시험 데이터 반복 사용을 완전히 막지는 못합니다. 현재 결과는 이미지 단위 점추정이며 환자/그룹 bootstrap 신뢰구간과 미학습 제품 거절 성능은 아직 없습니다. Top-3는 클래스가 3개 이하일 때 정보가 거의 없으므로 K도 함께 보고합니다. support가 0인 제품의 recall은 null이며, 전체 클래스 macro-F1에서는 F1=0을 포함합니다.

## 다음 개발 순서

현장 JSON·환자 그룹·제품 수를 확인하고 기준모델을 한 번 완주한 뒤, calibration 기반 확률 보정과 거절 기준, 그룹 bootstrap 신뢰구간, 품질 변화에 대한 강건성 평가를 추가합니다. 미학습 제품 실험은 제외할 제품을 먼저 확정하고 별도 학습·평가 설계를 마련해야 합니다. 현재 기준모델 점수를 미학습 제품 탐지 성능으로 해석하지 않습니다.

GitHub에는 코드·합성 테스트·빈 양식만 올립니다. 실제 설정, 분할, 사례별 예측, 학습 가중치, 그림을 포함한 모든 분석 산출물은 안심존의 검토 절차를 따릅니다.
