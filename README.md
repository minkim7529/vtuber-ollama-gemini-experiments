# 로컬 LLM 보이스 컴패니언 — 실제 변경 내역

[Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber) (v1.2.1) 위에서
로컬(Ollama)·클라우드(Gemini) 백엔드를 오가며 실험하고, 소형 로컬 모델의 실패
유형에 대응하기 위해 직접 작성한 코드입니다.

업스트림 오픈소스 프로젝트 전체를 이 저장소에 다시 올리는 대신, **직접 작성/수정한
부분만** `git diff` 형태로 남겼습니다.

## 파일

- [`my-changes.patch`](my-changes.patch) — 실제 코드 변경 전체 (`git diff` 원본,
  `uv.lock` 제외)
- [`diff-summary.txt`](diff-summary.txt) — 변경된 파일과 줄 수 요약

## 변경 요약 (15개 파일, +500 / -9줄)

| 파일 | 내용 |
|---|---|
| `agent/stateless_llm/ollama_llm.py` | **신규 파일(252줄)** — Ollama 네이티브 `/api/chat` 백엔드. 스트리밍, 이미지 다운스케일, `<think>` 태그 스트리핑, 한자/이모지 필터, 모델 preload/unload 생명주기 |
| `agent/transformers.py` | 정체성 역전 감지·치환 데코레이터(`identity_guard`) |
| `agent/agents/basic_memory_agent.py` | 위 데코레이터를 응답 파이프라인에 연결, temperature 실험 관련 주석 |
| `conversations/conversation_handler.py` | 능동 발화(proactive-speak) 상황 변형 5종 + 최근 발화 반복 방지 로직 |
| `tts/edge_tts.py`, `tts/tts_factory.py` | Edge TTS(무료 티어)로 교체하며 rate/pitch 파라미터 노출 |
| `config_manager/stateless_llm.py`, `config_manager/tts.py` | 위 변경들에 대응하는 설정 스키마 추가 |
| `agent/stateless_llm/openai_compatible_llm.py` | OpenAI 호환 경로(Gemini 등) 관련 수정 |

원본 코드 위치: `git diff`를 실행한 로컬 저장소는
`Open-LLM-VTuber/Open-LLM-VTuber`의 `v1.2.1` 릴리스를 그대로 클론한 뒤 위 파일들만
수정한 상태입니다.
