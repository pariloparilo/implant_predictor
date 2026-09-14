# Implant predictor

K-Health 본선을 위한 폐쇄망 임플란트 영상 분석 코드입니다. **크롭된 영상의 제조사·모델 식별**을 목표로 합니다. 현재 v0.2는 영상–JSON 매칭, 데이터 감사, 그룹 분할 확정, ResNet50 기준모델 학습·재개, 제품별/병원별/촬영방법별 평가와 그림 생성을 지원합니다.

실제 의료데이터로 검증한 모델이나 임상용 제품은 아닙니다. 미학습 제품 거절, 확률 보정, 신뢰구간, Grad-CAM은 다음 개발 단계이며 현재 구현에 포함하지 않습니다.

## 첫 방문: 새로 확인된 제공 폴더 구조

2026-09-14 확인된 `1.Training / 2.Validation`과 `원천데이터/h01~h15` 구조는 새 `visit` 명령으로 점검합니다. **먼저 [첫 방문 실행 카드](FIRST_VISIT_QUICKSTART.md)를 사용하세요.** [현장 실행 계획](FIELD_VISIT_PLAN.md)과 [방문 후 전달 양식](FIELD_FEEDBACK_TEMPLATE.md)에 실행·관찰·후속 결정 항목을 정리했습니다. 결과 파일을 외부에 보내지 않아도 허용된 구조 설명·상태로 다음 개발을 결정할 수 있습니다.

기존 60/15/10/15 분할 예시는 현장 원분할·그룹키 확인 전 확정안이 아닙니다. visit 설정은 점검 전용이며 기존 train 명령에 직접 전달할 수 없습니다.

## 시작하기

Python 3.10 이상을 사용합니다. 인터넷이 가능한 **로컬 개발환경의 별도 가상환경**에서만 다음을 실행합니다.

```bash
python -m pip install -r requirements-train.txt
python -m unittest discover -s implant_analysis/tests -v
IMPLANT_RUN_ML_TESTS=1 python -m unittest implant_analysis.tests.test_learning -v
```

안심존에서는 기관이 승인한 기설치 패키지·가중치를 사용합니다. 코드가 패키지나 모델을 자동 다운로드하지 않습니다. Ubuntu/CUDA 신청 버전은 `requirements-train-cu126.txt`에 있으며, 현장에서 인터넷 설치용으로 사용하지 않습니다.

1. [데이터 감사·분할 안내](implant_analysis/README.md)를 따라 100개 샘플에서 매칭과 JSON 필드를 확인합니다.
2. 환자 또는 원촬영 그룹키를 기관에 확인하고 전체 감사를 진행합니다. `일련번호`를 환자 ID로 자동 간주하지 않습니다.
3. 분할 초안과 제품 범위를 검토한 뒤 [학습·평가 안내](TRAINING.md)를 따릅니다.

감사·분할만 사용할 경우 `requirements-audit.txt`의 Pillow만 있으면 됩니다. ML 패키지는 학습 명령을 실행할 때 불러옵니다.

## 팀 작업 규칙

1. 기능별 브랜치에서 수정하고 테스트·검토 후 합칩니다.
2. 실제 JSON·영상·환자 대응표·실행 결과·모델 가중치는 **비공개 GitHub에도 커밋하지 않습니다.**
3. 실제 설정은 `implant_analysis/configs/site.json`에만 둡니다. 공유용 `site.example.json`에는 실제 경로·식별자를 넣지 않습니다.
4. CSV 양식에는 열 이름만 두고, 테스트 데이터는 코드가 생성한 합성 영상만 사용합니다.
5. `.gitignore`를 우회하는 강제 추가를 하지 않고 커밋 전 파일 목록을 확인합니다.

## 검증 범위

기존 데이터 감사 테스트와 분할 누수·지표 계산 테스트, 별도 실행하는 실제 ResNet50 CPU 합성 테스트를 제공합니다. 학습 테스트는 학습→epoch 단위 재개→평가→그림 생성, 재개와 연속 실행의 일치, 가중치 해시 오류·분할/영상 변경 차단을 검사합니다. 합성 테스트 성능은 연구 성능으로 사용하지 않습니다.

로컬 macOS ARM64 검증과 안심존 Ubuntu 20.04/CPython 3.10/CUDA 실행은 다릅니다. 현장에서 GPU·패키지·반입 가중치의 최초 실행 검증이 필요합니다. `.gitignore`는 반출 승인이나 비식별화 도구가 아닙니다.
