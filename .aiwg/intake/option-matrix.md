# Option Matrix (Project Context & Intent)

**Purpose**: Capture what this project IS - its nature, audience, constraints, and intent - to determine appropriate SDLC framework application (templates, commands, agents, rigor levels).

**Generated**: 2025-12-04

---

## Step 1: Project Reality

### What IS This Project?

**Project Description** (in natural language):

A command-line tool for extracting audio from video files (MKV format) and transcribing audio in common formats (MP3, AAC, FLAC, etc.) using OpenAI Whisper API. Built by a small team (2-10 people) over 1-3 months for internal team productivity. Primary goal is 80% team adoption within 2 months, streamlining transcription workflows that currently require manual multi-step processes (extract audio manually, upload to service, download transcript).

### Audience & Scale

**Who uses this?** (check all that apply)
- [ ] Just me (personal project)
- [x] Small team (2-10 people, known individuals)
- [ ] Department (10-100 people, organization-internal)
- [ ] External customers (100-10k users, paying or free)
- [ ] Large scale (10k-100k+ users, public-facing)
- [ ] Other:

**Audience Characteristics**:
- Technical sophistication: [x] Technical (engineering team members)
- User risk tolerance: [x] Expects stability (internal tool for daily productivity)
- Support expectations: [x] Best-effort (internal tool, team self-support, GitHub Issues for bugs)

**Usage Scale** (current or projected):
- Active users: 2-10 initially (engineering team), 10-20 in 6 months (expanded to other departments)
- Request volume: Variable (ad-hoc transcription needs, batch processing of meeting recordings/interviews)
- Data volume: N/A (local file processing, no centralized storage, audio files remain on user's filesystem)
- Geographic distribution: [x] Single location (US-based team)

### Deployment & Infrastructure

**Expected Deployment Model** (what will this become?):
- [x] Client-only (CLI tool, local execution)
- [ ] Static site
- [ ] Client-server
- [ ] Full-stack application
- [ ] Multi-system
- [ ] Distributed application
- [ ] Embedded/IoT
- [ ] Hybrid
- [ ] Other:

**Where does this run?**
- [x] Local only (user's laptop/desktop, not deployed to server)
- [ ] Personal hosting
- [ ] Cloud platform
- [ ] On-premise
- [ ] Hybrid
- [ ] Edge/CDN
- [ ] Mobile
- [ ] Desktop
- [ ] Browser
- [ ] Other:

**Infrastructure Complexity**:
- Deployment type: [x] Client-only (pip installable, runs locally)
- Data persistence: [x] File system (audio files and transcripts on user's local machine)
- External dependencies: 1 third-party service (OpenAI Whisper API for transcription)
- Network topology: [x] Client-server (CLI → OpenAI API)

### Technical Complexity

**Codebase Characteristics**:
- Size: [x] 1k-10k LoC (estimated for CLI tool with extraction, transcription, formatting modules)
- Languages: Python 3.9+ (primary), Shell scripts (secondary for FFmpeg integration)
- Architecture: [x] Modular (CLI entry, extraction module, transcription module, formatter, batch processor, config manager)
- Team familiarity: [x] Greenfield (starting fresh, new project)

**Technical Risk Factors** (check all that apply):
- [ ] Performance-sensitive (latency, throughput critical)
- [ ] Security-sensitive (PII, payments, authentication)
- [ ] Data integrity-critical (financial, medical, legal records)
- [ ] High concurrency (many simultaneous users/processes)
- [x] Complex business logic (audio format detection, file chunking for API limits, batch processing, resume support)
- [x] Integration-heavy (OpenAI Whisper API, FFmpeg external dependency)
- [ ] None (straightforward technical requirements)

---

## Step 2: Constraints & Context

### Resources

**Team**:
- Size: 2-5 developers (initial development), 2-10 users total (team members who will use the tool)
- Experience: [x] Mixed (strong Python and API integration, learning FFmpeg audio/video processing)
- Availability: [x] Part-time (20-40% capacity, other priorities in parallel)

**Budget**:
- Development: [x] Moderate (part-time team allocation, 1-3 month timeline)
- Infrastructure: $5-20/month (OpenAI Whisper API usage, ~10-50 hours of audio transcribed per month at $0.006/minute)
- Timeline: 1-3 months to MVP (polished MVP for team rollout)

### Regulatory & Compliance

**Data Sensitivity** (check all that apply):
- [ ] Public data only (no privacy concerns)
- [ ] User-provided content (email, profile, preferences)
- [ ] Personally Identifiable Information (PII: name, address, phone)
- [ ] Payment information (credit cards, financial accounts)
- [ ] Protected Health Information (PHI: medical records)
- [x] Sensitive business data (team-generated audio: meeting recordings, technical interviews, lectures)
- [ ] Other:

**Note**: Audio files and transcripts are team-generated content, remain on user's local filesystem (no centralized storage), processed temporarily by OpenAI API (per Whisper API terms, not retained). Internal tool, not public-facing.

**Regulatory Requirements** (check all that apply):
- [x] None (no specific regulations, internal tool, no PII storage, team-generated content only)
- [ ] GDPR
- [ ] CCPA
- [ ] HIPAA
- [ ] PCI-DSS
- [ ] SOX
- [ ] FedRAMP
- [ ] ISO27001
- [ ] SOC2
- [ ] Other:

**Contractual Obligations** (check all that apply):
- [x] None (no contracts, internal tool)
- [ ] SLA commitments
- [ ] Security requirements
- [ ] Compliance certifications
- [ ] Data residency
- [ ] Right to audit
- [ ] Other:

### Technical Context

**Current State** (for existing projects):
- Current stage: [x] Concept (new project, planning phase before Sprint 1)
- Test coverage: Target 60% (MVP with quality focus)
- Documentation: Target basic (README with installation, usage, config, troubleshooting)
- Deployment automation: Target CI/CD basic (GitHub Actions for linting, tests, security scans)

**Technical Debt** (for existing projects):
- Severity: [x] None (new/clean, greenfield project)
- Type: N/A
- Priority: N/A

---

## Step 3: Priorities & Trade-offs

### What Matters Most?

**Rank these priorities** (1 = most important, 4 = least important):
- `2` Speed to delivery (1-3 month timeline, but not urgent)
- `3` Cost efficiency (budget-conscious, API costs are low)
- `4` Quality & security (team adoption depends on reliability and ease of use)
- `1` Reliability & scale (team adoption is primary success metric)

**Note**: User indicated "Team adoption" as primary success metric (80% adoption goal). This maps to Quality/Reliability priority (tool must work well to drive adoption).

**Priority Weights** (must sum to 1.0, derived from ranking):

| Criterion | Weight | Rationale |
|-----------|--------|-----------|
| **Delivery speed** | 0.25 | 1-3 month timeline is reasonable, not rushed. Can iterate post-MVP. Faster than typical internal tool development. |
| **Cost efficiency** | 0.15 | API costs are low ($5-20/month), infrastructure is minimal (local CLI tool). Budget is not a primary constraint. |
| **Quality/security** | 0.30 | Team adoption depends on quality (reliability, ease of use, clear error messages). 60% test coverage target reflects quality focus. Baseline security (API key management, dependency scanning) is appropriate. |
| **Reliability/scale** | 0.30 | Primary success metric is 80% team adoption. Tool must "just work" for meeting recordings, batch processing, large files. Success rate >95% for valid files. Graceful error handling critical. |
| **TOTAL** | **1.00** | ← Sum verified |

### Trade-off Context

**What are you optimizing for?** (in your own words)

Team adoption is the primary goal (80% adoption within 2 months). This requires a tool that is:
1. **Easy to use**: Single command execution (`transcribe audio.mp3` or `transcribe video.mkv --extract-audio`)
2. **Reliable**: Successfully process 95%+ of files without errors, graceful error handling, clear troubleshooting messages
3. **Time-saving**: Reduce transcription workflow from ~30 minutes (manual extraction, upload, download) to <5 minutes (automated)
4. **Well-documented**: Clear installation guide (FFmpeg setup across platforms), usage examples, configuration options

Optimizing for **quality and reliability** to drive adoption, with moderate speed (1-3 months is reasonable for polished MVP).

**What are you willing to sacrifice?** (be explicit)

1. **Initial test coverage**: Start with 30% coverage (core logic), expand to 60% as features stabilize. Unit tests for critical paths (extraction, API integration), defer comprehensive e2e tests.
2. **Advanced features**: Defer to post-MVP:
   - Speaker identification (Whisper API supports, but added complexity)
   - AI-generated summaries (requires additional OpenAI GPT API calls, cost and complexity)
   - VTT and JSON output formats (prioritize txt and SRT for MVP, add VTT/JSON in v2)
   - Real-time transcription (streaming audio, significant architectural complexity)
3. **Perfect documentation**: Start with basic README (installation, usage, troubleshooting), expand based on user feedback. Defer API docs (Sphinx-generated) to post-MVP.
4. **PyPI publishing**: Start with local installation (`pip install -e .`) for team testing, defer public PyPI package to v1.0.
5. **Performance optimization**: Target <6 minutes for 1-hour MKV → transcript workflow. Optimize further if users report pain points.

**What is non-negotiable?** (constraints that override trade-offs)

1. **Ease of use**: Single command execution with clear progress indicators (rich library progress bars). If users find it complex, adoption will fail.
2. **Error handling**: Must handle common failure modes gracefully:
   - FFmpeg not installed → clear error message with installation instructions
   - API rate limits → retry with exponential backoff, clear guidance to user
   - Unsupported file formats → format detection, conversion fallback, helpful error
   - Large file failures → chunking, resume support, progress checkpointing
3. **API key security**: Never hardcode API keys, clear documentation on secure practices (environment variables, .env file, restrictive permissions). Exposure risk is moderate (cost, abuse potential).
4. **Batch processing**: Core feature for team workflows (transcribe entire folders of meeting recordings). Must work reliably with progress tracking.
5. **Large file support**: Files >1GB (2+ hour recordings) are common use case. Must handle via chunking, not fail outright.

---

## Step 4: Intent & Decision Context

### Why This Intake Now?

**What triggered this intake?** (check all that apply)
- [x] Starting new project (need to plan approach)
- [ ] Documenting existing project
- [ ] Preparing for scale/growth
- [ ] Compliance requirement
- [ ] Team expansion
- [ ] Technical pivot
- [ ] Handoff/transition
- [ ] Funding/business milestone
- [x] Other: Seeking SDLC structure (want organized development process with AIWG framework)

**Context**: Team currently lacks efficient transcription workflow. Manual process is time-consuming (extract audio manually, upload to online service, download transcript). This intake establishes clear requirements, architecture, and success criteria before development starts to ensure:
- Feature scope is well-defined (avoid scope creep)
- Team alignment on technical approach (Python + FFmpeg + Whisper API)
- Risks identified early (FFmpeg installation, large file handling, API rate limits)
- Structured SDLC process (Inception → Elaboration → Construction → Transition)

**What decisions need making?** (be specific)

1. **Architecture: FFmpeg integration approach**
   - Option A: Use `ffmpeg-python` library (Python wrapper, abstracts FFmpeg commands)
   - Option B: Direct subprocess calls to FFmpeg binary (more control, less abstraction)
   - **Decision needed**: Which provides better error handling and flexibility for format detection?

2. **Batch processing: Async strategy**
   - Option A: Python `asyncio` for concurrent API calls (I/O-bound operations)
   - Option B: `concurrent.futures` ThreadPoolExecutor (simpler, fewer edge cases)
   - **Decision needed**: Which is more maintainable for team with moderate async experience?

3. **Output formats: MVP scope**
   - Must-have: Plain text (.txt), timestamped SRT (subtitle format)
   - Nice-to-have: VTT (web video captions), JSON (structured data), speaker identification, AI summaries
   - **Decision needed**: Which formats are critical for team adoption vs. v2 features?

4. **Testing strategy: Coverage targets**
   - Initial: 30% (core logic only, fast iteration)
   - Target: 60% (comprehensive unit + integration tests)
   - **Decision needed**: What coverage threshold should block PR merges? 30% or 60%?

5. **Error handling: FFmpeg installation**
   - Option A: Startup check with error message (simple, user must install FFmpeg)
   - Option B: Bundle FFmpeg binaries with CLI tool (complex licensing, platform-specific builds)
   - **Decision needed**: Is user-installed FFmpeg acceptable, or must it "just work" out-of-box?

**What's uncertain or controversial?** (surface disagreements)

1. **Test coverage expectations**: Balance between quality (60% target for team adoption) and speed (30% initially to ship faster). Team values quality, but timeline is 1-3 months. What's the right threshold for MVP?

2. **FFmpeg learning curve**: Team has strong Python/API skills but limited FFmpeg/audio processing experience. Uncertainty around:
   - How complex is audio extraction from MKV? (might be 1 sprint, might be 2)
   - What codec edge cases will we encounter? (corrupted files, unusual formats)
   - Should we allocate 1-2 sprints for PoC/spike before committing to full development?

3. **API cost management**: Whisper API pricing is reasonable ($0.006/minute), but batch processing of many files could accumulate costs. Uncertain:
   - Should we implement cost tracking/warnings?
   - Should we cache transcripts to avoid re-processing duplicate files?
   - What's the cost ceiling before users hesitate to use the tool?

4. **Feature prioritization**: User selected all output formats (txt, SRT, VTT, JSON, speaker ID, summaries) in interactive questions. Scope may be ambitious for 1-3 month timeline. Need to decide:
   - What's truly critical for v1? (txt + SRT likely sufficient)
   - What can defer to v2 based on user feedback?

**Success criteria for this intake process**:

1. **Clear technical direction**: Team aligned on architecture (Python + FFmpeg + Whisper API), with ADRs documenting key decisions.
2. **Realistic timeline and scope**: Well-defined MVP features (extraction, transcription, batch, txt + SRT output) achievable in 1-3 months.
3. **Risk mitigation plan**: FFmpeg installation, large file handling, API rate limits addressed with specific strategies.
4. **Structured development process**: AIWG framework components identified (templates, commands, agents) appropriate for small team and MVP profile.
5. **Measurable success**: 80% team adoption within 2 months, 70% time savings vs. manual workflow.

---

## Step 5: Framework Application

### Relevant SDLC Components

Based on project reality (Step 1) and priorities (Step 3), which framework components are relevant?

**Templates** (check applicable):
- [x] Intake (project-intake, solution-profile, option-matrix) - **Always include**
- [x] Requirements (user-stories, use-cases, NFRs) - Include: small team (2-5 developers), need shared understanding of features, coordination for 1-3 month timeline
- [x] Architecture (SAD, ADRs, API contracts) - Include: modular design (6 components), integration-heavy (FFmpeg, Whisper API), technical decisions needed (FFmpeg approach, async strategy)
- [x] Test (test-strategy, test-plan, test-cases) - Include: 60% coverage target, quality-critical for team adoption, integration testing (FFmpeg, API)
- [ ] Security (threat-model, security-requirements) - Skip: Baseline security sufficient (API key management, dependency scanning), no PII storage, internal tool
- [x] Deployment (deployment-plan, runbook, ORR) - Include: README for installation (FFmpeg setup), runbook for troubleshooting (common errors, API issues)
- [ ] Governance (decision-log, CCB-minutes, RACI) - Skip: Small team (2-5 developers), informal coordination, no change control overhead needed

**Commands** (check applicable):
- [x] Intake commands (intake-wizard, intake-start) - **Always include**
- [x] Flow commands (iteration, discovery, delivery) - Include: /flow-iteration-dual-track for 2-week sprints, ongoing development over 1-3 months
- [ ] Quality gates (security-gate, gate-check, traceability) - Skip: MVP profile, no compliance requirements, lightweight process
- [x] Specialized (build-poc, pr-review, troubleshooting-guide) - Include: /build-poc for FFmpeg spike (validate extraction approach), /pr-review for code quality

**Agents** (check applicable):
- [x] Core SDLC agents (requirements-analyst, architect, code-reviewer, test-engineer, devops) - Include: structured process for small team, multi-agent workflow for artifact generation
- [ ] Security specialists (security-gatekeeper, security-auditor) - Skip: Baseline security, no compliance
- [ ] Operations specialists (incident-responder, reliability-engineer) - Skip: CLI tool (not a service), best-effort support
- [ ] Enterprise specialists (legal-liaison, compliance-validator, privacy-officer) - Skip: No regulatory requirements, internal tool

**Process Rigor Level**:
- [ ] Minimal (README, lightweight notes, ad-hoc)
- [x] Moderate (user stories, basic architecture, test plan, runbook) - For: Small team (2-5), MVP timeline (1-3 months), internal tool with quality focus
- [ ] Full (comprehensive docs, traceability, gates)
- [ ] Enterprise (audit trails, compliance evidence, change control)

### Rationale for Framework Choices

**Why this subset of framework?**

Audio Transcription CLI Tool (MVP, small team, 1-3 months) needs **Moderate rigor**:

1. **Intake**: Establish baseline, align team on priorities and architecture (this document + project-intake + solution-profile)

2. **Requirements**: User stories for core features (extraction, transcription, batch processing, output formats). Small team (2-5 developers) benefits from shared understanding to coordinate work over 1-3 month timeline.

3. **Architecture**: Basic component diagram, ADRs for critical decisions:
   - ADR-001: Python + FFmpeg over all-in-one library (rationale: flexibility, industry-standard)
   - ADR-002: Async batch processing with asyncio (rationale: I/O-bound API calls, concurrency)
   - ADR-003: Multiple output formats (txt, SRT, VTT, JSON) - which for MVP vs v2?

   Architecture docs help team understand modular design (CLI entry, extraction, transcription, formatter, batch processor, config manager).

4. **Testing**: Test plan covering critical paths (extraction, API integration, batch processing, error handling). 60% coverage target requires strategy for what to test first (unit tests for business logic, integration tests for workflows).

5. **Deployment**: README with installation (Python, FFmpeg), usage examples, configuration (.env setup). Runbook for troubleshooting:
   - FFmpeg not found → platform-specific installation instructions
   - API rate limits → retry behavior, concurrency limits
   - Unsupported formats → conversion options, format detection

6. **Core SDLC agents**: Use AIWG multi-agent workflow for efficient artifact generation:
   - Requirements Analyst: User stories for features
   - Architecture Designer: Component diagram, ADRs
   - Test Engineer: Test strategy, coverage plan
   - Code Reviewer: PR reviews for quality

7. **Flow commands**: /flow-iteration-dual-track for 2-week sprints (Discovery + Delivery in parallel)

8. **Specialized commands**: /build-poc for FFmpeg extraction spike (validate approach before full implementation)

**What we're skipping and why**:

1. **Security templates**: Skip threat-model, security-requirements, incident-response
   - **Reason**: Baseline security sufficient (API key management, dependency scanning), no PII storage, internal tool
   - **Revisit trigger**: If audio files contain PII (customer interviews), upgrade to Strong security posture

2. **Governance templates**: Skip decision-log (formal), CCB-minutes, RACI, change-control
   - **Reason**: Small team (2-5 developers), informal coordination, no enterprise overhead needed
   - **Revisit trigger**: If team grows >5 people or external stakeholders added

3. **Quality gates**: Skip /security-gate, /gate-check, /check-traceability
   - **Reason**: MVP profile, no compliance requirements, lightweight process
   - **Revisit trigger**: If compliance requirements emerge or tool becomes mission-critical

4. **Operations specialists**: Skip incident-responder, reliability-engineer
   - **Reason**: CLI tool (not a service), best-effort support, no SLA
   - **Revisit trigger**: If tool becomes critical to team workflows (zero-tolerance for downtime)

5. **Enterprise specialists**: Skip legal-liaison, compliance-validator, privacy-officer
   - **Reason**: No regulatory requirements, internal tool, no contracts
   - **Revisit trigger**: If open-sourced or commercial use case emerges

---

## Step 6: Evolution & Adaptation

### Expected Changes

**How might this project evolve?**
- [ ] No planned changes (stable scope and scale)
- [x] User base growth (when: 6 months, trigger: tool proves valuable, expanded to other departments 10-20 users)
- [x] Feature expansion (when: 3-6 months post-MVP, trigger: user feedback identifies high-value additions)
- [x] Team expansion (when: if open-sourced or commercialized, trigger: external contributors or customers)
- [ ] Commercial/monetization (potential future: if tool proves valuable externally)
- [ ] Compliance requirements (potential future: if audio files contain PII/sensitive content)
- [x] Technical pivot (when: 6-12 months, trigger: add local Whisper model support for offline use, reduce API costs)

**Adaptation Triggers** (when to revisit framework application):

1. **Add security templates when PII introduced** (timeline: month 4 if user accounts with customer interview audio planned)
   - Trigger: Audio files contain customer PII (interviews, support calls)
   - Action: Upgrade to Strong security (threat model, SAST/DAST, data handling controls)

2. **Add governance templates when team exceeds 5 people** (timeline: hiring planned post-Series A if commercialized)
   - Trigger: Team grows beyond small team coordination capacity
   - Action: Add decision-log, RACI, change control for stakeholder alignment

3. **Upgrade to Production profile when beta ends** (timeline: month 6, trigger: 80% team adoption validated)
   - Trigger: Tool becomes critical to team workflows, 500+ active users (if expanded organization-wide)
   - Action: Increase test coverage to 80%, add APM monitoring, create comprehensive runbooks, SLO tracking

4. **Add compliance templates when requirements emerge** (timeline: if audio contains sensitive content)
   - Trigger: GDPR/CCPA requirements (EU/California customers in audio), HIPAA (healthcare audio)
   - Action: Add privacy-officer agent, compliance-validator, data handling policies

**Planned Framework Evolution**:

- **Current (Inception - Month 1)**:
  - Intake (project-intake, solution-profile, option-matrix)
  - Requirements (user stories for core features)
  - Architecture (component diagram, 3 ADRs)
  - Test plan (60% coverage strategy)
  - README (installation, usage, config)
  - Runbook (basic troubleshooting)

- **3 months (Elaboration - MVP Complete)**:
  - Expand requirements (user stories for v2 features based on feedback)
  - Update ADRs (document architectural learnings)
  - Expand test suite (60% coverage achieved, integration tests with real API)
  - Comprehensive runbook (based on early user feedback and common issues)
  - User feedback loop (survey team after 1 month of usage)

- **6 months (Construction - Production Rollout if successful)**:
  - Upgrade to Production profile if validated (80% adoption achieved)
  - Add security templates if PII introduced (customer interview audio)
  - Add governance if team expands (>5 people, external stakeholders)
  - Performance optimization (if usage patterns reveal bottlenecks)
  - Comprehensive SAD (full software architecture document)

- **12 months (Transition/Production - Mature Tool)**:
  - If open-sourced: Add contribution guidelines, community docs
  - If commercialized: Add enterprise features (SSO, multi-tenancy, commercial support)
  - If compliance required: Add SOC2/ISO27001 templates
  - Technical pivot: Local Whisper model support (offline mode, cost reduction)

---

## Architectural Options Analysis

### Option A: Python CLI + FFmpeg Library Wrapper (ffmpeg-python)

**Description**: Use `ffmpeg-python` library to wrap FFmpeg commands. Library provides Pythonic interface to FFmpeg, abstracts command construction, handles common patterns.

**Technology Stack**:
- Python 3.9+ (click/typer CLI framework)
- ffmpeg-python library (FFmpeg wrapper)
- OpenAI Python SDK (Whisper API)
- FFmpeg binary (external dependency, user-installed)

**Scoring** (0-5 scale):

| Criterion | Score | Rationale |
|-----------|------:|-----------|
| Delivery Speed | 4/5 | Library abstracts FFmpeg complexity, faster development. Examples and docs available. Learning curve moderate (library API + FFmpeg concepts). |
| Cost Efficiency | 5/5 | Free library, no licensing costs. Same API costs as other options ($5-20/month). |
| Quality/Security | 4/5 | Library handles common edge cases, error handling. Potential risk: library bugs or FFmpeg version incompatibilities. Security: validates inputs, but relies on FFmpeg binary security. |
| Reliability/Scale | 4/5 | Proven library (used in production systems). FFmpeg is industry-standard. Edge cases: unusual codecs may require direct FFmpeg calls anyway. |
| **Weighted Total** | **4.15** | (4 × 0.25) + (5 × 0.15) + (4 × 0.30) + (4 × 0.30) = 1.0 + 0.75 + 1.2 + 1.2 = 4.15 |

**Trade-offs**:
- **Pros**: Pythonic interface (easier for team), abstracts FFmpeg command construction, handles common patterns (format conversion, codec detection), faster development
- **Cons**: Added dependency (library maintenance risk), abstracts away control (may need direct FFmpeg calls for edge cases), learning curve (library API + FFmpeg concepts)

**When to choose**: Best for team with strong Python skills but limited FFmpeg experience. Prioritizes development speed and maintainability.

### Option B: Python CLI + Direct FFmpeg Subprocess Calls

**Description**: Use Python `subprocess` module to call FFmpeg binary directly with command-line arguments. Full control over FFmpeg commands, no library abstraction.

**Technology Stack**:
- Python 3.9+ (click/typer CLI framework)
- subprocess module (standard library, no dependencies)
- OpenAI Python SDK (Whisper API)
- FFmpeg binary (external dependency, user-installed)

**Scoring** (0-5 scale):

| Criterion | Score | Rationale |
|-----------|------:|-----------|
| Delivery Speed | 3/5 | Slower initial development (must learn FFmpeg command syntax, build commands manually). More testing needed for edge cases (escape args, error parsing). |
| Cost Efficiency | 5/5 | No library dependency (stdlib only). Same API costs ($5-20/month). |
| Quality/Security | 3/5 | Full control over commands, but more prone to errors (command construction bugs, argument escaping). Security: must manually sanitize inputs, validate file paths. More attack surface for injection. |
| Reliability/Scale | 4/5 | Direct FFmpeg control handles edge cases better (unusual codecs, complex filters). But more code to maintain (error parsing, retry logic). |
| **Weighted Total** | **3.70** | (3 × 0.25) + (5 × 0.15) + (3 × 0.30) + (4 × 0.30) = 0.75 + 0.75 + 0.9 + 1.2 = 3.60 |

**Trade-offs**:
- **Pros**: Full control over FFmpeg (no library limitations), no external dependency (stdlib only), direct debugging (see exact FFmpeg commands), handles edge cases better
- **Cons**: Slower development (manual command construction, error parsing), more code to maintain (retry logic, input validation), security risks (command injection if not careful), steeper learning curve (FFmpeg syntax)

**When to choose**: Best for team with FFmpeg experience or when library abstraction is limiting. Prioritizes control and minimal dependencies over development speed.

### Option C: Python CLI + All-in-One Audio Library (pydub)

**Description**: Use `pydub` library for audio extraction and manipulation. High-level Python API for audio operations, wraps FFmpeg internally but provides simpler interface.

**Technology Stack**:
- Python 3.9+ (click/typer CLI framework)
- pydub library (audio processing, FFmpeg wrapper)
- OpenAI Python SDK (Whisper API)
- FFmpeg binary (external dependency, pydub uses it internally)

**Scoring** (0-5 scale):

| Criterion | Score | Rationale |
|-----------|------:|-----------|
| Delivery Speed | 5/5 | Simplest API (one-liners for audio extraction). Minimal FFmpeg knowledge required. Fastest development. |
| Cost Efficiency | 5/5 | Free library. Same API costs ($5-20/month). |
| Quality/Security | 3/5 | Simplified API hides complexity (good for MVP). But limited control for edge cases (unusual codecs, complex operations). Security: relies on pydub and FFmpeg. |
| Reliability/Scale | 3/5 | Great for common formats (MP3, WAV, AAC). Limited for edge cases (rare codecs, advanced filters). May hit pydub limitations and need FFmpeg anyway. |
| **Weighted Total** | **3.85** | (5 × 0.25) + (5 × 0.15) + (3 × 0.30) + (3 × 0.30) = 1.25 + 0.75 + 0.9 + 0.9 = 3.80 |

**Trade-offs**:
- **Pros**: Simplest API (one-liner audio extraction), fastest development, minimal FFmpeg knowledge needed, great for common formats
- **Cons**: Limited control (high-level abstraction), edge case handling uncertain (may still need FFmpeg for rare codecs), potential pydub limitations (may hit ceiling and need to switch later)

**When to choose**: Best for rapid MVP with common audio formats (MP3, WAV, AAC). Risk of hitting limitations later (may need FFmpeg for MKV edge cases).

---

## Recommendation

**Recommended Option**: **Option A: Python CLI + FFmpeg Library Wrapper (ffmpeg-python)** (Score: 4.15/5.0)

**Rationale**: Best balance for small team and MVP timeline:

1. **Development speed** (weight 0.25, score 4/5): Library abstracts FFmpeg complexity while providing Pythonic interface team is comfortable with. Faster than direct subprocess calls (Option B), slightly slower than pydub (Option C) but more flexible.

2. **Quality** (weight 0.30, score 4/5): Library handles common edge cases and error handling. Team values quality for adoption (60% test coverage target). Option A provides good error handling without building everything from scratch.

3. **Reliability** (weight 0.30, score 4/5): Proven library used in production. Handles MKV extraction (core requirement) and various codecs (AAC, MP3, FLAC). FFmpeg is industry-standard, so reliability is high.

4. **Flexibility for edge cases**: Unlike pydub (Option C), ffmpeg-python allows dropping down to direct FFmpeg calls if needed. Best of both worlds: Pythonic API for common cases, FFmpeg power for edge cases.

5. **Team skills**: Strong Python skills, learning FFmpeg concepts. Library provides gradual learning curve (start with high-level API, learn FFmpeg as needed).

**Sensitivities**:

- **If timeline pressure increases** (compress to <2 months): Consider Option C (pydub) for fastest MVP, accept risk of hitting limitations later.
- **If edge cases dominate early testing** (unusual codecs, corrupted files): Reconsider Option B (direct FFmpeg) for maximum control, accept slower development.
- **If library maintenance becomes an issue** (ffmpeg-python unmaintained, bugs): Fallback to Option B (direct subprocess) to remove dependency.

**Implementation Plan**:

1. **Sprint 1-2: PoC with ffmpeg-python**
   - Install ffmpeg-python, test audio extraction from sample MKV files
   - Validate codec support (AAC, MP3, FLAC embedded in MKV)
   - Test error handling (corrupted files, unsupported codecs)
   - Decision point: If PoC succeeds, proceed with Option A. If major blockers, pivot to Option B.

2. **Sprint 3-4: Core extraction module**
   - Build extraction module using ffmpeg-python API
   - Format detection (magic bytes, not extensions)
   - Error handling (missing FFmpeg, unsupported codecs, corrupted files)
   - Unit tests for extraction logic

3. **Sprint 5-6: Integration with Whisper API**
   - Integrate OpenAI SDK for transcription
   - File chunking for >25MB files (API limit)
   - Async batch processing with asyncio
   - Integration tests (full workflow: MKV → extract → transcribe → txt/SRT)

**Risks and Mitigations**:

- **Risk 1**: ffmpeg-python library limitations (unusual codecs, complex operations)
  - **Mitigation**: Library allows dropping down to direct FFmpeg calls via `.run()` method. Hybrid approach: Use library for 80% of cases, direct calls for edge cases.

- **Risk 2**: FFmpeg version incompatibilities (library tested on older FFmpeg)
  - **Mitigation**: Document minimum FFmpeg version (4.0+), test on team's platforms (Linux, macOS, Windows). Pin library version in requirements.txt.

- **Risk 3**: Team unfamiliarity with FFmpeg concepts (codecs, filters, formats)
  - **Mitigation**: Allocate 1-2 sprints for PoC/learning. Reference FFmpeg documentation and ffmpeg-python examples. Pair programming for knowledge sharing.

---

## Next Steps

1. **Review option-matrix and validate priorities align** with team/stakeholder expectations (speed, cost, quality, reliability weights)
2. **Confirm architectural recommendation** (Option A: ffmpeg-python) with technical leads
3. **Start PoC** (Sprint 1-2): Validate ffmpeg-python for MKV extraction with sample files
4. **Use recommended framework components** from Step 5 for Inception phase:
   - Requirements Analyst: User stories for core features
   - Architecture Designer: Component diagram, ADR-001 (FFmpeg approach)
   - Test Engineer: Test strategy with 60% coverage plan
5. **Revisit framework selection at phase gates** (Inception → Elaboration → Construction → Transition)
6. **Track evolution triggers** (PII introduced, team expansion, compliance requirements) and adapt framework as needed
