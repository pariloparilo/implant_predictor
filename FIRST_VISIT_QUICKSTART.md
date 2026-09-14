# 첫 방문 실행 카드

목표: 환경과 제공 Training/Validation을 확인하고, **학습 분할을 정할 근거**를 확보합니다. 자세한 설계는 FIELD_VISIT_PLAN.md를 참고합니다.

## 0. 설정

반입 승인된 프로젝트 폴더에서 `implant_analysis/configs/visit.example.json`을 `visit.json`으로 복사하고, 다음 두 값을 우선 수정합니다.

- data_root: 실제 `01.데이터` 폴더의 절대경로
- work_dir: 기관이 허용한 별도 결과 폴더의 절대경로. 원본 데이터 내부로 설정하지 않음

`provider_partitions`의 `1.Training/라벨링데이터`, `1.Training/원천데이터`, `2.Validation/라벨링데이터`, `2.Validation/원천데이터`가 실제 표기와 맞는지 확인합니다. `training.weights_path/weights_sha256`은 승인된 파일 경로와 전체 해시가 있을 때 입력합니다. 없다면 null로 두고 데이터 점검부터 진행할 수 있습니다. pairing/grouping의 verified는 현장 검토 전 false를 유지합니다.

## 1. 환경·경로 점검

```bash
python -m implant_analysis visit --config implant_analysis/configs/visit.json --run-id day1_preflight --stage preflight --check-ml
```

실행이 끝나면 `work_dir/visit/day1_preflight/READ_ON_SCREEN.txt`를 엽니다. HTML을 열 수 있는 환경이면 같은 폴더의 `dashboard.html`을 사용합니다. 인터넷·외부 글꼴·JavaScript가 필요 없습니다. CUDA_UNAVAILABLE이면 GPU가 없는지 드라이버/패키지 문제가 있는지 담당자에게 확인합니다. 설치 버전은 대회 신청·실제 승인 버전과 비교합니다.

## 2. 분산 표본 점검

```bash
python -m implant_analysis visit --config implant_analysis/configs/visit.json --run-id day1_sample --stage sample
```

기본 각 제공 영역 150개를 상위 h폴더에 분산해 점검합니다. `work_dir/visit/day1_sample/dashboard.html` 또는 READ_ON_SCREEN.txt를 열고 오류·누락을 확인합니다. `contact_sheet.png`는 내부 크롭 검토 화면이며 최대 30개를 보여줍니다. 번호별 대응은 `visual_review_internal.csv`에서 확인합니다.

수동 확인: JSON과 영상 파일 대응, 단일 임플란트 크롭 여부, 끝부분 잘림, 방향·반전, 큰 문자/표식·화살표·불필요 여백. 품질이 낮다는 이유만으로 자동 제외하지 않고 관찰을 기록합니다. 표본 결과가 정상이어도 전체 통과로 해석하지 않습니다.

## 3. 전체 감사

매칭 규칙·필드 구조를 먼저 확인한 뒤 실행합니다. grouping을 확보하지 못해도 이 단계의 데이터 감사는 가능합니다.

```bash
python -m implant_analysis visit --config implant_analysis/configs/visit.json --run-id day1_full --stage full
```

소요시간에 여유를 둡니다. 중단되면 완료한 것으로 취급하지 않으며 새 run-id로 다시 실행합니다. `COMPLETED_WITH_ERRORS`는 일부 실패입니다. `summary.json`에 RUNNING이 남은 실행은 완료하지 못한 실행입니다.

전체 요약: `work_dir/visit/day1_full/READ_ON_SCREEN.txt`.

상세 원인은 해당 실행 아래 `internal/audit/provided_train/` 또는 `internal/audit/provided_validation/`의 issues.csv, manifest.csv, summary.json에 있습니다. 두 영역에 걸친 정확한 중복은 실행 폴더의 cross_partition_duplicates.csv로 확인합니다. 이 코드는 중복 파일을 삭제하거나 분할을 바꾸지 않습니다.

## 4. 귀가 전

FIELD_FEEDBACK_TEMPLATE.md의 A항목과 기관 답변을 확인합니다. 원본 JSON·영상·케이스 대응표·미리보기·예측표·가중치 파일은 내부에 둡니다. 구두/메모 전달이 허용된 항목만 설명할 수 있도록 기록합니다. 실행 성공만으로 train을 진행하지 않고, 제공 분할 기준과 그룹키 결과에 따라 다음 학습 연결을 결정합니다.

프로그램 자체가 시작되지 않으면: 설치된 Python 버전, `python -m implant_analysis --help` 실행 여부, 발생한 오류 종류를 확인합니다. Pillow가 없으면 기승인 패키지 설치 상태를 운영 측에 확인합니다. 모든 명령은 패키지를 다운로드하지 않습니다.
