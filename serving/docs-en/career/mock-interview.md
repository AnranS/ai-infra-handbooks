# Mock interviews: a scoring rubric and retrospectives

<p class="lead">However many problems you practice, until you answer them under time pressure with someone following up, you don't know whether you really know them. This chapter gives a mock interview you can organize yourself: how to set the questions, how to score, how to review. Start once you have worked through the inference engine systematically, with at least one session a week.</p>

## The structure of one mock interview {#一场模拟面试的结构}

Arranged over 60 minutes, matching the four most common segments of a real interview:

| Segment | Length | Questions | What it tests |
| --- | --- | --- | --- |
| Coding | 20 minutes | 1 algorithm problem (medium), or implementing 1 component from scratch | whether you can write it correctly within the time, edge cases and complexity |
| Fundamentals follow-ups | 15 minutes | draw 3 questions from the bank, with 2–3 levels of follow-up each | depth of principles, and whether you can give numbers |
| System design | 15 minutes | 1 system design problem | requirements → estimates → architecture → the critical path → failures and scaling |
| Project deep dive | 10 minutes | your own project or open-source contributions | difficulties, numbers, trade-offs, failed attempts |

The interviewer can be a colleague, a study partner, or a large model playing the role (the prompt is [below](#让大模型当面试官)). Record the whole thing and play it back when reviewing; listening to yourself once beats anyone's feedback.

Four sets of questions arranged to this structure are in [mock interview sets](mock-exams.md).

## The scoring rubric {#评分表}

Four items, 1–4 points each. Score on **behavior**, not impressions:

| Score | Correctness | Depth | Numbers | Communication |
| --- | --- | --- | --- | --- |
| 1 | clear errors, unnoticed | can only restate concepts | no numbers | no structure, saying whatever comes to mind |
| 2 | errors, corrected after a hint | can explain the principle but fails the second follow-up | has numbers, but the magnitude is wrong or the source can't be named | structured but long-winded, needing interruption |
| 3 | basically correct with only minor flaws | handles two levels of follow-up (why, what it costs) | the key numbers are right and can be estimated on the spot | conclusions first then details, with the interviewer able to follow |
| 4 | fully correct, proactively stating premises and boundaries | reaches implementation details and alternatives, and says under what conditions the conclusion changes | estimates fast and accurately, arguing trade-offs with numbers | proactively confirms requirements, draws diagrams, controls the pace |

**The bar: 3 or above on all four.** Only three sessions in a row at the bar count as ready. For the coding segment, also record two things: whether all tests passed within the time, and how many hints were used.

## A list of question sources {#出题清单}

- **Coding**: [implementation problems](coding.md), and the corresponding problems in [the practice bank](root://practice/) (most with automatic grading);
- **Fundamentals follow-ups**: this handbook's [inference interview question bank](interview.md), the LLM handbook's [self-test bank](llm://synthesis/quiz/), the CUDA handbook's [interview question bank](cuda://career/interview/), and the [estimation bank](root://practice/#/?q=估算) in the practice problems;
- **System design**: [system design problems](system-design.md), after first working through [estimating deployment scale](root://practice/#/p/sv-est-cluster-size);
- **Project deep dive**: prepare 3 difficulties, 3 numbers and 1 failed attempt per project.

Three common ways to follow up, which an interviewer can use straight off the page to find the depth:

1. **Why not...**: "why a radix tree instead of hashed blocks?" "why not tensor parallelism?"
2. **What does it cost**: "what did chunked prefill make worse?" "what are the risks of an FP8 KV Cache?"
3. **What are the numbers / what happens at ten times the scale**: "how many milliseconds is one decode step?" "going from 8K to 128K of context, what breaks first?"

## Having a large model play the interviewer {#让大模型当面试官}

With no one to practice with, use the prompt below to have a large model play the interviewer (replacing what's in brackets):

<!-- i18n:diagram dbf6d5aca9 -->
```text
You are a senior interviewer in large-model inference systems, interviewing a candidate for an inference framework engineering role.
This round covers: [fundamentals follow-ups: KV Cache management and scheduling / system design: an online inference service for a 600B-class MoE model / project deep dive: ...].

Rules:
1. Ask one question at a time and wait for my answer before continuing;
2. Follow up 2-3 levels deep on my answer: why do it this way, what does it cost, what are the specific numbers, what happens at ten times the scale;
3. When I am wrong or vague, don't correct me directly; keep following up and let me expose the problem myself;
4. After 15 minutes (about 6-8 exchanges), finish and score 1-4 on each of "correctness, depth, numbers, communication",
   giving the specific reason for each deduction, and finally list the questions I answered poorly along with the correct key points.
```

Large models score leniently, so double-check numeric questions yourself; where they help most is "following up tirelessly".

## A retrospective template {#复盘模板}

Within 24 hours after every mock interview (and every real one), review with the same template:

<!-- i18n:diagram b024a15e80 -->
```markdown
## Mock interview #N (date)

- Interviewer / format:
- Scores: correctness _ / depth _ / numbers _ / communication _
- Coding: problem, time taken, whether it passed, number of hints

### Questions I answered poorly

| Question | What I said | Where the problem was | The correct key points | Chapters / exercises to review |
| --- | --- | --- | --- | --- |

### One thing to change next time
```

"Where the problem was" has only three answers: **didn't know** (go read the chapter), **knew but couldn't say it** (write a one-sentence answer onto a flashcard), and **knew but didn't think of it** (do more timed practice).
Tell which one it is, and the next step is clear.

## Common ways to lose points {#常见的失分点}

- **Concepts only, no numbers**: nearly every question in inference systems can come down to bandwidth, compute or memory, and giving the magnitude is the biggest bonus;
- **Not asking about requirements before designing**: drawing an architecture right away. First confirm QPS, input and output lengths, the SLO and cost constraints, then estimate, and only then the architecture;
- **Describing only what you did on a project**: interviewers want to hear why it was hard, how you located it, how much it helped, and what still isn't good;
- **Bluffing when a follow-up goes past your knowledge**: stating clearly what you are sure of, then giving your reasoning and how you would verify it, beats inventing an answer by far.
