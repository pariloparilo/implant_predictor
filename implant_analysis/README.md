> v0.2: 아래는 데이터 감사·분할 설명입니다. 추가된 학습·평가 실행 순서는 [TRAINING.md](../TRAINING.md)를 따르세요.

# 임플란트 분석 코드 v0.1: 자료 감사와 분할 초안

이 버전은 실제 학습 전에 필요한 영상-JSON 매칭, 데이터 감사, 중복 점검, 그룹 분할 초안을 구현합니다. 학습·추론·Energy/MSP 평가 코드는 아직 포함하지 않습니다. 실제 의료데이터로 실행한 결과가 아니며, PDF에 나온 필드와 합성 자료로 검증했습니다.

## 자료 구조와 확인한 필드

```text
데이터루트/
  라벨링데이터/병원별데이터/
    병원별데이터_01/*.json
    병원별데이터_02/*.json
  원시데이터/병원별데이터/
    병원별데이터_01/*.jpeg
    병원별데이터_02/*.jpeg
```

두 루트 아래를 재귀 탐색하므로 폴더별로 코드를 복사하지 않습니다. 원자료에는 쓰지 않으며 결과는 별도 work 폴더에 저장합니다. 기본 매칭은 **각 루트 아래의 상대경로에서 확장자만 제거한 값이 같은 경우**입니다. 예를 들어 `병원별데이터_01/A.json`과 `병원별데이터_01/A.jpeg`입니다. 실제 파일명이 이 규칙을 따르는지는 현장에서 확인해야 합니다. 정렬 순서·일련번호·전역 파일명만으로 임의 연결하지 않습니다.

설명자료 PDF 6~8쪽의 필드는 다음과 같습니다.

| JSON 필드 | 코드의 표준 열 | 용도 |
|---|---|---|
| 일련번호 | serial | 감사용; 환자 ID로 사용하지 않음 |
| 촬영방법 | modality | 촬영종류별 분석 |
| 제조사 | manufacturer | 정답 라벨 구성 |
| 모델명 | model | 정답 라벨 구성 |
| 직경 / 길이 | diameter / length | 메타데이터 감사용 |
| 식립위치 | tooth_position | 메타데이터 감사용 |
| 식립일자 / 촬영일자 | implant_date / capture_date | 메타데이터 감사용 |
| 촬영병원 | hospital | 기관별 집계; 폴더 번호와 구분 |

영상 식별기의 입력은 크롭 영상만 사용하도록 후속 학습기를 작성합니다. 제조사·모델명 외 메타데이터를 분류 입력으로 섞지 않습니다. 라벨은 `[제조사, 모델명]` 조합으로 만들고 class_id는 분할 단계에서 안정적으로 부여합니다. `TS II`/`TS III` 또는 모델명 뒤 괄호를 자동 통합하지 않습니다. 검토한 별칭만 별도 CSV로 반영합니다. 숫자와 날짜의 원문은 보존하며 이 버전에서 임상적 타당성을 검증하지 않습니다.

## 첫 실행

Python 3.10 이상과 Pillow가 필요합니다. 이 단계는 torch·GPU 없이 실행합니다. 새 패키지를 자동 설치하거나 네트워크에 접속하는 코드는 없습니다.

1. 기관 절차에 따라 코드와 설정을 반입합니다.
2. `configs/site.example.json`을 `configs/site.json`으로 복사합니다.
3. `data_root`를 실제 데이터 루트로 바꿉니다. 결과 폴더는 `work_dir`에서 정합니다.
4. `implant_analysis` 폴더의 상위 디렉터리에서 실행합니다.

```bash
python -m implant_analysis audit --config implant_analysis/configs/site.json --run-id day1_sample --limit 100
```

기본 설정에서는 `implant_analysis/work/audit/day1_sample/`에 결과가 생깁니다. 이 실행은 정렬된 첫 100개 JSON의 기능 점검이며 대표성 있는 표본조사가 아닙니다. 부분 감사도 전체 영상의 파일 목록은 읽지만 영상 디코딩은 선택된 JSON에 연결된 파일만 수행합니다.

먼저 `summary.json`, `issues.csv`, `manifest.csv`를 확인합니다. 매칭·JSON 구조가 맞으면 전체 감사를 실행합니다.

```bash
python -m implant_analysis audit --config implant_analysis/configs/site.json --run-id day1_full
```

같은 run-id 결과를 덮어쓰지 않습니다. 수정 후 재실행은 `day1_full_v2`처럼 새 이름을 사용합니다. 중단된 실행의 자동 재개 기능은 아직 없으며, 부분 폴더가 남으면 새 run-id로 다시 실행합니다. 모든 영상 해시 계산과 디코딩은 저장장치 속도에 따라 시간이 걸리므로 100개 점검에서 처리량부터 확인하세요.

## 감사 결과

| 파일 | 내용 |
|---|---|
| manifest.csv | JSON 1개당 1행; 영상 경로·라벨·크기·메타데이터·해시·그룹 ID·상태 |
| summary.json | 파일 수, 사용 가능 수, 클래스 수, JSON 구조, 필드 결측, 오류 수 |
| issues.csv | 매칭 실패·모호한 매칭·손상·라벨 충돌·영상 모드 점검 대상 |
| unmatched_images.csv | 연결된 JSON을 찾지 못한 영상; 부분 감사에서는 미검사 영상도 포함 |
| duplicate_pixels.csv | 디코딩 픽셀이 완전히 같은 영상과 라벨 충돌 여부 |
| class_hospital_modality_counts.csv | 제조사·모델·촬영병원·촬영종류·상태별 영상 수 |
| config_snapshot.json | 실행 시 사용한 설정 |

깨진 영상, JSON 구조 오류, 제조사/모델 누락, 모호한 연결, 같은 영상에 상충하는 라벨은 제외 대상으로 기록합니다. 원자료는 삭제하지 않습니다. L/RGB 이외의 영상 모드는 별도 검토 전까지 제외합니다. 간단한 픽셀 표준편차는 기술적 지표이며 저품질 임상 정답이 아닙니다.

정확한 중복 검사는 파일 해시와 디코딩된 RGB 픽셀 해시를 사용합니다. 재압축으로 픽셀이 변한 영상, 유사한 크롭, 반복 촬영은 이 검사만으로 발견하지 못합니다. 실제 그룹키가 필요한 이유입니다.

## 파일명이나 JSON 구조가 다를 때

- 이름이 다르면 `templates/pairing.csv` 형식으로 `label_relpath,image_relpath`를 만들고 pairing.mode를 `mapping_csv`, pairing.csv를 해당 경로로 설정합니다. 폴더 경계 밖의 파일은 연결하지 않습니다.
- JSON이 감싸진 객체라면 `record_path`를 실제 구조에 맞춰 설정합니다. 예: `{"annotation": {...}}`이면 `["annotation"]`. 여러 레코드 중 하나를 임의로 선택해서는 안 됩니다.
- PDF와 필드명이 다르면 `fields`만 수정합니다. 모르는 구조를 추측해서 평탄화하지 않습니다.
- 잘못된 JSON 인코딩은 오류로 기록합니다. 실제 인코딩을 확인한 뒤 `json_encoding`을 변경합니다.
- JSON 내용이나 환자정보가 포함된 파일을 외부에 보내는 것을 전제하지 않습니다. 현장에서 구조·키 이름·파일명 규칙을 확인해 설정을 조정하세요.

## 환자·원촬영 그룹키가 확인된 뒤

PDF의 일련번호는 '병원별로 부여된 번호'이고 환자 식별자라고 명시되지 않았습니다. 또한 `병원별데이터_01`이라는 이름만으로 특정 병원이라고 판단하지 않습니다. 날짜·식립위치·직경 조합을 임의 환자키로 만들지 않습니다.

기관이 의미를 확인한 그룹키가 있으면 `templates/groups.csv` 형식으로 **안심존 내부에서** 대응표를 작성합니다.

```text
label_relpath,group_key,source_image_key
병원별데이터_01/A.json,기관이_확인한_그룹키,기관이_확인한_원촬영키
```

- `group_key`: 학습과 평가 사이에 섞이면 안 되는 단위의 키. 동일 환자 또는 동일 원촬영은 동일 키를 사용합니다.
- `source_image_key`: 원촬영에서 여러 크롭이 나온 경우 동일한 키를 사용합니다. 확인되지 않으면 비워 둡니다.
- 키는 기관·폴더 사이에서도 구분되는 범위로 제공되어야 합니다. 단순 번호 재사용에 주의하고, 병원 간 같은 환자의 연결 범위도 확인합니다.
- `grouping.kind`는 환자 확인 시 `patient`, 원촬영만 확인 시 `source_image`입니다.
- `grouping.csv`를 지정하고 실제 검토가 끝난 경우에만 `grouping.verified`와 `pairing.verified`를 true로 바꿉니다. `verification_note`에 확인 근거를 기록합니다.
- 변경한 설정으로 전체 감사를 새로 실행해야 합니다. 분할은 감사 당시의 확인 상태를 사용합니다.

```bash
python -m implant_analysis audit --config implant_analysis/configs/site.json --run-id grouped_full
python -m implant_analysis split-draft --config implant_analysis/configs/site.json --audit-dir implant_analysis/work/audit/grouped_full --run-id split_v1
```

부분 감사, 확인되지 않은 매칭/그룹키, 그룹키 누락, 변경된 감사 manifest로는 분할하지 않습니다. 그룹 정보를 확보할 수 없다면 여기서 이미지 무작위 분할로 우회하지 말고, 기관별 평가 등 별도 검증 설계를 먼저 검토해야 합니다.

## 분할은 초안이며 검토 후 동결

동일 환자/사례, 원촬영, 정확한 중복 영상의 연결 관계를 합쳐 한 집합에 넣습니다. 연결 관계를 만든 후 같은 픽셀·같은 라벨의 복제본은 하나만 남기고 제거 대응표를 기록합니다.

목표 비율은 train/dev/calibration/test = 60/15/10/15입니다. 그룹 전체를 이동하기 때문에 영상 수 비율이 정확히 일치하지 않습니다. 네 집합이 비지 않게 그룹 수를 배분하고, 고정된 64개 후보 중 라벨·그룹 수의 균형만 보고 초안을 선택합니다. 모델 성능이나 영상 특징은 후보 선택에 사용하지 않습니다.

결과는 `split_manifest.csv`, `class_map.json`, `class_split_counts.csv`, `removed_exact_duplicates.csv`, `summary.json`입니다. 모두 `DRAFT_REQUIRES_REVIEW_BEFORE_TRAINING` 상태로 생성합니다.

- `train_missing_classes`가 비어 있는지 확인합니다. 비어 있지 않으면 전체 클래스 식별 학습을 바로 시작할 수 없습니다.
- 클래스별 영상 수와 연결 그룹 수를 확인합니다. 기본 주평가 후보 기준은 train20/dev5/calibration10/test20 그룹이며 운영 기준일 뿐입니다.
- 희소 클래스를 조용히 버리지 않습니다. 주평가·탐색·학습 전용의 범위를 검토하고 기록합니다.
- 최초 학습 전에 분할, 평가 대상, 미학습 홀드아웃 제품 목록을 동결합니다. 결과가 나쁜 제품을 사후 제외하지 않습니다.

## 구현 상태와 후속 실험

v0.2에는 분할 확정(`study.py`), ResNet50 학습·재개·평가(`learning.py`), 지표(`metrics.py`), 그림 생성(`figures.py`)이 추가되었습니다. 실행 방법은 [TRAINING.md](../TRAINING.md)를 따르세요. 기준모델은 전체 제품 closed-set 분류입니다.

미학습 홀드아웃 제품 실험, MSP/Energy 거절 기준, 확률 보정, 그룹 신뢰구간은 아직 구현하지 않았습니다. 해당 실험을 추가할 때 제외 제품은 학습 전에 확정하고, ImageNet 가중치에서 별도 학습을 시작해야 합니다. 전체 제품을 학습한 체크포인트로 제외 제품 실험을 시작하지 않습니다.

## 검증 및 한계

```bash
python -m unittest discover -s implant_analysis/tests -v
```

데이터 감사·분할의 기존 12개 합성 테스트 항목은 다음과 같습니다: UTF-8 BOM·일련번호 보존, 모호한/누락 매칭, 명시적 대응표, 손상 영상, 중복 JSON 키, 중복 영상의 상충 라벨, 그룹키·전체 감사 선행 조건, 그룹·원촬영·중복 관계의 분할 누수 차단, 재현성, 결과 덮어쓰기 및 원자료 폴더 쓰기 방지 등입니다.

추가된 분할 검증·지표·ResNet50 CPU 합성 통합 테스트를 포함한 검증 범위는 [VALIDATION.md](../VALIDATION.md)에 기록했습니다. Python3.10 구문 검사는 수행했지만 실제 안심존 Ubuntu/Python3.10/CUDA 검증은 별도입니다. 실제 데이터 매칭, 환자키 의미, 대규모 처리시간은 현장에서 확인해야 합니다. 공식 사전학습 가중치는 다운로드하지 않았고, 실제 의료데이터 학습은 수행하지 않았습니다.

모든 결과는 기본적으로 안심존 내부 전용입니다. 파일 경로·날짜·기관명·그룹 해시는 반출 승인 없이 외부 공유하지 않습니다. 해시를 붙였다는 이유로 익명화가 완료된 것은 아닙니다.
