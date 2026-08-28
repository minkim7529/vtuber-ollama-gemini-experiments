# 로컬 LLM 보이스 컴패니언 — Ollama ↔ Gemini 백엔드 실험

[Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber) (v1.2.1) 위에서
로컬(Ollama)·클라우드(Gemini) LLM 백엔드를 바꿔가며 답변 품질을 비교해보고,
소형 로컬 모델(7B급)이 자주 보이는 실패 유형에 대응하는 코드를 직접
작성했습니다. 업스트림 오픈소스 전체를 다시 올리는 대신, **직접 작성/수정한
부분만** `git diff` 형태로 남겼습니다.

## 배경 — 왜 두 백엔드를 비교했나

완전히 무료로, 로컬에서 돌아가는 컴패니언을 만드는 게 목표였습니다. 같은
캐릭터 프롬프트를 그대로 둔 채 LLM만 Ollama(로컬 7B급)와 Gemini(무료 티어
클라우드) 사이에서 바꿔 끼우며 비교해봤습니다.

- **Ollama**: 프롬프트를 아무리 고쳐써도 정확도·자연스러움이 어느 선을
  못 넘었습니다. 아래에 정리한 정체성 역전·한자 혼입·반복 발화 문제도
  결국 이 한계에서 나온 증상이었습니다.
- **Gemini**: 같은 프롬프트에서 훨씬 정확하고 자연스러운 답변이 나왔지만,
  무료 토큰/요청 한도 때문에 오래 반복 실험하기는 어려웠습니다.

결국 오케스트레이션 레이어(프롬프트/후처리)를 아무리 다듬어도 기반 LLM의
능력 자체가 상한선이라는 걸 확인했고, 그 다음부터는 **그 상한선 안에서
로컬 모델의 품질을 최대한 끌어올리는** 쪽으로 코드를 작성했습니다.

## 1. Ollama 네이티브 백엔드 (`ollama_llm.py`, 신규 252줄)

OpenAI 호환 엔드포인트 대신 Ollama의 네이티브 `/api/chat`을 직접 씁니다.

```python
class OllamaLLM(AsyncLLM):
    def __init__(self, model, base_url, ..., repeat_penalty: float = 1.1, top_p: float = 0.9):
        self.native_url = base_url.replace("/v1", "") + "/api/chat"
        ...
        # preload model — Ollama 서버가 안 떠 있으면 여기서 바로 알려줌
        requests.post(self.native_url, json={"model": model, "keep_alive": keep_alive})
        if unload_at_exit:
            atexit.register(self.cleanup)

    async def chat_completion(self, messages, system=None, tools=NOT_GIVEN, temperature_override=None):
        """
        도구 호출이 없으면 네이티브 /api/chat으로 "think": false를 보내
        Qwen3 계열의 추론 트레이스를 확실히 끈다 (OpenAI 호환 엔드포인트는
        이 옵션을 무시함). 답변 하나당 150여 추론 토큰을 절약.
        도구가 있으면 OpenAI 호환 경로로 폴백하되, <think> 블록은 안전망으로
        한 번 더 스트립한다.
        """
        if tools is NOT_GIVEN or not tools:
            async for chunk in self._strip_stray_hanzi(
                self._native_chat_completion(messages, system, temperature_override)
            ):
                yield chunk
            return
        async for chunk in self._strip_stray_hanzi(
            self._strip_thinking(super().chat_completion(messages, system, tools))
        ):
            yield chunk
```

**이미지 다운스케일.** 비전 모델에 풀HD 스크린샷을 그대로 보내면 인코딩이
오래 걸려서, 긴 변 1024px로 줄여서 보냅니다:

```python
def _downscale_base64_image(b64_data: str) -> str:
    raw = base64.b64decode(b64_data)
    with Image.open(io.BytesIO(raw)) as img:
        if max(img.size) <= MAX_IMAGE_DIMENSION:
            return b64_data
        img = img.convert("RGB")
        img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
```

**스트리밍 중 한자/이모지 제거.** 양자화 모델이 가끔 흘리는 CJK 한자,
전각 문자, 금지된 이모지를 청크 단위로 걸러냅니다. 청크가 항상 온전한
코드포인트라 청크를 넘나드는 버퍼링은 필요 없었습니다:

```python
HANZI_RE = re.compile(r"[一-鿿　-〿＀-￯]+")
EMOJI_RE = re.compile("[\U0001f000-\U0001faff\U00002600-\U000027bf...]+")

@staticmethod
async def _strip_stray_hanzi(source):
    async for item in source:
        if not isinstance(item, str):
            yield item
            continue
        yield EMOJI_RE.sub("", HANZI_RE.sub("", item))
```

**`<think>` 태그 스트리퍼.** 태그가 청크 경계에서 잘려도 안전하게 처리하려고
버퍼 기반 파서로 짰습니다 (열림/닫힘 태그 길이만큼 버퍼를 항상 유지합니다):

```python
@staticmethod
async def _strip_thinking(source):
    in_think = False
    buffer = ""
    max_tag_len = max(len(THINK_OPEN_TAG), len(THINK_CLOSE_TAG))
    async for item in source:
        if not isinstance(item, str):
            yield item
            continue
        buffer += item
        visible = ""
        while True:
            if not in_think:
                idx = buffer.find(THINK_OPEN_TAG)
                if idx == -1:
                    keep_from = max(0, len(buffer) - (max_tag_len - 1))
                    visible += buffer[:keep_from]
                    buffer = buffer[keep_from:]
                    break
                visible += buffer[:idx]
                buffer = buffer[idx + len(THINK_OPEN_TAG):]
                in_think = True
            else:
                idx = buffer.find(THINK_CLOSE_TAG)
                if idx == -1:
                    buffer = buffer[max(0, len(buffer) - (max_tag_len - 1)):]
                    break
                buffer = buffer[idx + len(THINK_CLOSE_TAG):]
                in_think = False
        if visible:
            yield visible
```

## 2. 정체성 역전 가드 (`transformers.py`)

로컬 7B 모델이 가끔 캐릭터-유저 관계를 반대로 말하는 문제가 있었습니다
(예: "내가 너의 오빠야", "당신은 하루의 여동생이에요"). 이걸 정규식으로 잡아
안전한 대사로 바꿔치기합니다. 문장이 다 끝난 시점(TTS 직전)에 검사해서,
완결된 문장만 봅니다.

```python
IDENTITY_REVERSAL_RE = re.compile(
    r"(내가|나는|난)\s*(너의|당신의|니)?\s*오빠"
    r"|(너는|넌|너가|네가|당신은|당신이)\s*.{0,15}?여동생"
)

def identity_guard():
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async for item in func(*args, **kwargs):
                if isinstance(item, SentenceWithTags) and IDENTITY_REVERSAL_RE.search(item.text):
                    logger.warning(f"identity_guard caught reversed-identity sentence: {item.text!r}")
                    item.text = random.choice(IDENTITY_GUARD_FALLBACKS)
                yield item
        return wrapper
    return decorator
```

## 3. 발화 다양성 엔진 (`conversation_handler.py`)

능동 발화(proactive-speak)마다 5가지 상황 변형 중 **직전과 겹치지 않는 것**을
뽑고, 최근 발화 3개를 기억해뒀다가 "이미 이렇게 말했으니 다르게 말해줘"를
프롬프트에 같이 넣어줍니다.

```python
PROACTIVE_SPEAK_VARIANTS = [
    "화면 이미지가 같이 전달됐다면 user가 지금 뭘 하고 있는지 보고 자연스럽게 참견해줘.",
    "user의 안부를 묻거나, 문득 user 생각이 났다는 듯 말을 걸어줘.",
    "오늘 있었던 사소한 생각이나 농담처럼, 갑자기 생각난 아무 말이나 툭 던져줘.",
    "user가 대답을 안 해준다고 살짝 삐진 티를 내줘.",
]

def _pick_proactive_variant(client_uid: str) -> str:
    """random.choice만 쓰면 1/5 확률로 직전과 같은 변형이 뽑히고, 그 경우
    거의 항상 비슷한 내용이 나오는 걸 실측으로 확인 — 직전 인덱스를 제외한다."""
    last_index = _last_proactive_variant_index.get(client_uid)
    choices = [i for i in range(len(PROACTIVE_SPEAK_VARIANTS)) if i != last_index]
    index = random.choice(choices)
    _last_proactive_variant_index[client_uid] = index
    return PROACTIVE_SPEAK_VARIANTS[index]

def _build_proactive_avoid_repeats_note(client_uid: str) -> str:
    history = _recent_proactive_utterances.get(client_uid)
    if not history:
        return ""
    quoted = ", ".join(f'"{h}"' for h in history)
    return f'\n\n당신이 최근에 이미 이렇게 말했어: {quoted}\n이번엔 내용도 표현도 완전히 다르게, 절대 비슷하게 말하지 마.'
```

**금지 문구를 직접 인용하면 오히려 역효과가 났습니다.** "이 표현은 쓰지 마:
'X'"처럼 금지어를 그대로 인용하면, 이 7B 모델은 그 표현에 오히려 앵커링돼서
더 자주 뱉어냈습니다. 그래서 금지 대신 원하는 스타일만 긍정문으로 서술하는
쪽으로 바꿨습니다:

```python
PROACTIVE_SPEAK_STYLE_NOTE = (
    "\n\n짧은 문자 보내듯 자연스럽게, 상투적이지 않은 그때그때 다른 표현으로 말해줘. "
    "문장을 시작하는 방식도 매번 다르게 - 이름부터 부르고 시작할 때도 있고, "
    "이름 없이 바로 용건이나 감정부터 꺼낼 때도 있게 섞어줘."
)
```

## 4. 검증한 것 vs 버린 것 — temperature 실험

능동 발화의 표현 다양성을 늘리려고 temperature를 기본값보다 올리는 실험을
+0.3/+0.9와 +0.1/+0.8, 두 폭으로 해봤습니다. 낮은 폭에서도 이모지 열거나
문장이 무너지는 등 일관성만 떨어졌고, 반복이 줄어드는 효과는 기본
temperature와 통계적으로 별 차이가 없었습니다. 위험 대비 얻는 게 없다고
판단해서 되돌렸고, 같은 실험을 또 반복하지 않도록 코드 주석에 그대로
남겨뒀습니다.

## 5. TTS — 무료 티어로 전환

Edge TTS(무료)로 바꾸면서 rate/pitch를 설정값으로 뺐습니다. 이렇게 해두면
나중에 원하는 보이스 모델을 직접 커스텀해서 얹을 수 있습니다.

```python
class TTSEngine(TTSInterface):
    def __init__(self, voice="en-US-AvaMultilingualNeural", rate="+0%", pitch="+0Hz"):
        self.voice = voice
        self.rate = rate
        self.pitch = pitch

    def generate_audio(self, text, file_name_no_ext=None):
        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate, pitch=self.pitch)
        communicate.save_sync(file_name)
```

## 관찰한 실패 유형과 대응 — 요약

| 증상 | 원인 | 대응 |
|---|---|---|
| 정체성 역전 ("내가 너의 오빠야") | 프롬프트만으로는 긴 세션에서 관계 설정이 안정적으로 유지되지 않음 | 문장 완성 시점에 정규식 검출 → 안전 대사로 치환 |
| 한자/전각 문자 혼입 ("팀장那儿 갔다가") | 양자화 모델의 잘 알려진 아티팩트 | CJK 한자·전각 부호 대역을 스트림 청크마다 제거 |
| 발화 반복 (2턴 연속 "엄마랑 같이…") | 상황 변형 5종을 단순 랜덤 추출 시 20%가 직전과 동일 | 직전 인덱스를 제외한 랜덤 추출 |
| 금지 문구 명시가 역효과 | 금지어를 직접 인용하면 앵커링되어 더 자주 재생산 | 긍정문으로만 스타일 서술 |

## 파일

- [`my-changes.patch`](my-changes.patch) — 실제 코드 변경 전체 (`git diff` 원본, `uv.lock` 제외)
- [`diff-summary.txt`](diff-summary.txt) — 변경된 파일과 줄 수 요약

## 변경 파일 목록 (15개 파일, +500 / -9줄)

| 파일 | 내용 |
|---|---|
| `agent/stateless_llm/ollama_llm.py` | 신규 파일(252줄) — Ollama 네이티브 백엔드 |
| `agent/transformers.py` | `identity_guard` 데코레이터 |
| `agent/agents/basic_memory_agent.py` | 데코레이터 연결 + temperature 실험 주석 |
| `conversations/conversation_handler.py` | 발화 다양성 엔진 |
| `tts/edge_tts.py`, `tts/tts_factory.py` | Edge TTS rate/pitch 파라미터 |
| `config_manager/stateless_llm.py`, `config_manager/tts.py` | 설정 스키마 추가 |
| `agent/stateless_llm/openai_compatible_llm.py` | OpenAI 호환 경로(Gemini 등) 관련 수정 |

원본 코드는 `Open-LLM-VTuber/Open-LLM-VTuber`의 `v1.2.1` 릴리스를 그대로
클론해서, 위 파일들만 수정한 상태입니다.
