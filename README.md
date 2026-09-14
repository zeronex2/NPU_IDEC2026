# NPU_IDEC2026
경북대학교 IDEC 2026 창의 회로설계 챌린지
- MNIST 분류용 CNN 가속기를 ASAP7 공정으로 설계하는 대회
- 주어진 Baseline 코드를 수정해서 PPA 개선 

Weight 상수화를 통한 One-hot Code 활용
Fully-connected 및 CSA Based MAC 연산 간소화를 주제로 한 프로젝트(진행중)


## 주요 구현 내용

- Weight 상수화
- One-hot Code 활용한 FC 모듈 개선
- Conv1, Conv2 모듈 내 MAC 연산 간소화
  - 곱셈기 -> Shift, |Weight|로 치환
  - RCA -> Wallace 3:2 CSA + Kogge-Stone
- Pipelining

## 구현 과정(11단계)
- Claude, Cursor AI 활용한 RTL 구현
- Cursor Agent Mode 활용한 Synthesis 및 Report 파일 생성 자동화
- 각 단계별 PPA 변화는 report 참조

1. Baseline
2. Weight 상수화(localparam)
3. FC one-hot Code
4. conv1 출력 FF Pipelining
5. conv2 3단 곱 Pipelining
6. FC data_out FF Pipelining
7. conv2 CSA+KS(Kogge-Stone)
8. conv1 CSA+KS(Kogge-Stone)
9. FC wsel_r FF Pipelining
10. conv2 MID07 Pipelining
11. FC 48 -> 12 Pipelining

## Development Environment

- Verilog HDL
- OpenROAD(Linux)
- MobaXterm
- Cursor AI(Agent)
- Claude
- iverilog
