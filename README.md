# Code Escape the Trap: Multi-Shot Video Benchmark Pipeline

ECCV 제출을 위한 고도화된 비디오 생성 모델 벤치마크 및 평가 파이프라인입니다.

## 🚀 다른 서버에서 시작하기 (Quick Start)

새로운 서버에서 실험 환경을 구축하고 파이프라인을 실행하려면 아래 명령어를 차례대로 입력하세요.

### 1. 레포지토리 클론 및 이동
```bash
git clone https://github.com/merrybabyxmas/code_escape_the_trap.git
cd code_escape_the_trap
```

### 2. 자동 환경 세팅 (Conda & Dependencies)
```bash
bash setup_env.sh
```

### 3. 실험 실행
```bash
conda activate paper_env
python master_huge_pipeline.py
```

## 📊 주요 기능 및 방어 데이터
본 파이프라인은 리뷰어의 공격을 방어하기 위해 다음과 같은 데이터를 자동으로 생성합니다:
- **LPIPS Timeline (.csv):** 동적 컷 탐지의 정당성을 증명하는 프레임별 변화량 곡선.
- **Raw Feature Bank (.pt):** DINOv2/CLIP 임베딩 원본 (재실험 없는 Ablation Study 가능).
- **DSA Matrix:** 지시 이행력을 입증하는 Softmax 정렬 행렬.
- **Worst 10 Analysis:** 정성적 분석을 위한 자동 실패 사례 추출.

## 📁 디렉토리 구조
- `master_huge_pipeline.py`: 메인 실험 및 평가 엔진.
- `visualizer.py`: 논문용 그래프(Pareto Frontier, Heatmap 등) 생성기.
- `datasets/`: 1,000개(Track S/M) 프롬프트 데이터셋.
- `outputs/`: 생성된 비디오 및 평가 결과 저장소 (Git 관리 제외).

---
**Note:** `outputs/` 폴더 내의 비디오 파일(`.mp4`)은 용량 문제로 Git에서 제외되어 있습니다. 기존 서버의 비디오가 있다면 `outputs/` 폴더를 직접 복사해오면 생성 단계를 건너뛰고 **재평가만 즉시 수행**할 수 있습니다.
