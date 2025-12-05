# Risk List: Audio Transcription CLI Tool

---

## Document Information

| Attribute | Value |
|-----------|-------|
| **Version** | 1.0 |
| **Status** | BASELINED |
| **Date** | 2025-12-04 |
| **Project** | Audio Transcription CLI Tool |
| **Phase** | Inception |
| **Owner** | Project Manager |
| **Contributors** | Project Manager (Primary), Architecture Designer, Security Architect |

---

## Executive Summary

This consolidated Risk List identifies **12 unique risks** for the Audio Transcription CLI Tool project, synthesized from business, technical, and security assessments. All duplicate entries have been merged, and risks are prioritized by severity score.

**Risk Distribution:**
- **HIGH Priority** (Score >= 15): 3 risks
- **MEDIUM Priority** (Score 9-14): 7 risks
- **LOW Priority** (Score < 9): 2 risks
- **Show Stoppers**: None identified

**Top 3 Risks Requiring Immediate Attention:**
1. **RISK-001: Scope Creep** (Score: 20) - Highest risk, requires disciplined scope control
2. **RISK-002: FFmpeg Installation Barrier** (Score: 15) - Blocks 30-40% of users, especially Windows
3. **RISK-003: Large File Handling/Chunking** (Score: 15) - Complex implementation for key use case

All top 3 risks have detailed mitigation plans with clear ownership and monitoring criteria.

---

## Risk Summary Table

| ID | Risk | Category | Likelihood | Impact | Score | Priority | Status |
|----|------|----------|------------|--------|-------|----------|--------|
| RISK-001 | Scope Creep | Business | High (5) | High (4) | **20** | HIGH | Mitigating |
| RISK-002 | FFmpeg Installation Barrier | Technical | High (5) | Medium (3) | **15** | HIGH | Mitigating |
| RISK-003 | Large File Handling/Chunking | Technical | Medium (3) | High (4) | **15** | HIGH | Open |
| RISK-004 | Aggressive Timeline | Schedule | Medium (3) | High (4) | **12** | MEDIUM | Open |
| RISK-005 | Low Team Adoption | Business | Medium (3) | High (4) | **12** | MEDIUM | Open |
| RISK-006 | API Key Exposure | Security | Medium (3) | Medium (3) | **9** | MEDIUM | Open |
| RISK-007 | Command Injection via File Paths | Security | Low (2) | Medium (3) | **9** | MEDIUM | Open |
| RISK-008 | API Rate Limits / Batch Processing | Technical | Medium (3) | Medium (3) | **9** | MEDIUM | Open |
| RISK-009 | Audio Format Compatibility | Technical | Medium (3) | Medium (3) | **9** | MEDIUM | Open |
| RISK-010 | Dependency Vulnerabilities | Security | Medium (3) | Medium (3) | **9** | MEDIUM | Open |
| RISK-011 | Whisper API Pricing/Policy Changes | External | Low (2) | High (4) | **8** | LOW | Open |
| RISK-012 | Part-Time Team Availability | Resource | Medium (3) | Low (2) | **6** | LOW | Accepted |

**Risk Score Calculation**: Likelihood (High=5, Medium=3, Low=2, Very Low=1) x Impact (Show Stopper=5, High=4, Medium=3, Low=2)

---

## Detailed Risk Register

---

### RISK-001: Scope Creep

**Priority**: HIGH (Score: 20)
**Category**: Business
**Status**: Mitigating
**Owner**: Product Owner / Project Manager

#### Description

The 1-3 month MVP timeline with comprehensive feature set (audio extraction, transcription, batch processing, multiple output formats, large file handling) creates high risk of scope expansion. User requests for additional features (speaker ID, summaries, cloud integration, real-time transcription) during development could extend timeline significantly, delaying ROI and causing team frustration.

#### Risk Analysis

| Factor | Assessment | Rationale |
|--------|------------|-----------|
| **Likelihood** | High (5) | Users naturally request features once they see working prototype; No formal scope change control process defined; Ambitious feature list already in intake |
| **Impact** | High (4) | Timeline extension from 3 months to 6+ months delays productivity gains; Risk of never reaching "done" state; Delayed adoption while waiting for features |

#### Detailed Mitigation Plan

**1. Strict MVP Scope Definition (Week 1 - Inception)**
- Document non-negotiable MVP features: MKV extraction, audio transcription, TXT + SRT output
- Explicitly list deferred features for v2: speaker ID, summaries, VTT, JSON, cloud storage
- Get stakeholder sign-off on MVP scope before Construction phase
- Create "v2 Backlog" parking lot for future requests

**2. Ruthless Prioritization Framework**
- Every feature request evaluated against criteria:
  - **Must-Have**: Blocks 80% team adoption or core workflow
  - **Should-Have**: Enhances UX but not required for MVP
  - **Nice-to-Have**: Defer to v2 or later
- Product Owner has final authority on scope decisions
- Default answer: "Great idea for v2, let's validate MVP first"

**3. 2-Week Sprint Cadence with Gate Reviews**
- Sprint 0 (Weeks 1-2): Architecture + PoC
- Sprint 1 (Weeks 3-4): Audio extraction module
- Sprint 2 (Weeks 5-6): Transcription module
- Sprint 3 (Weeks 7-8): Batch processing + output formats
- Sprint 4 (Weeks 9-10): Large file handling + testing
- Sprint 5 (Weeks 11-12): Documentation + polish
- Each sprint ends with scope review: "Are we on track for 3-month target?"

**4. Feature Freeze Policy**
- Week 8 (end of Sprint 3): Feature freeze for MVP
- Weeks 9-12: Bug fixes, testing, documentation only
- Any new features automatically go to v2 backlog
- Exceptions require Product Owner approval + timeline extension acknowledgment

**5. Scope Change Control Process**
- Formalize lightweight change request process:
  - Requestor: Describe feature + business value
  - Product Owner: Evaluate MVP impact (timeline, complexity)
  - Decision: Accept (extend timeline), Defer (v2), or Reject
- Track all change requests in `.aiwg/planning/change-log.md`

#### Monitoring

| Frequency | Activity | Owner |
|-----------|----------|-------|
| Weekly | Review active work items vs. original Sprint plan; Flag scope additions | Dev Team |
| Bi-Weekly | Assess cumulative scope drift; Velocity check | Project Manager |
| Monthly | Report scope status to Engineering Team Lead | Product Owner |

#### Success Metrics
- Zero unauthorized scope additions (all changes approved by Product Owner)
- MVP delivered within 3 months (+/- 2 weeks acceptable)
- v2 backlog contains 10+ deferred features (evidence of discipline)

#### Contingency Plan
- If timeline extends beyond 4 months: Conduct scope cut (remove batch processing or large file handling from MVP)
- If team pressure for features is high: Schedule "v2 Planning Session" to acknowledge and roadmap requests

#### Escalation
- If scope creep threatens 6+ month timeline: Escalate to Engineering Team Lead for executive decision

---

### RISK-002: FFmpeg Installation Barrier

**Priority**: HIGH (Score: 15)
**Category**: Technical
**Status**: Mitigating
**Owner**: Tech Lead / Developer

*Consolidated from: Primary Draft RISK-003, Technical Risks TECH-006*

#### Description

FFmpeg is a required external dependency for audio extraction from MKV files and format conversion. Many users, especially on Windows, may not have FFmpeg installed or properly configured in their system PATH. Installation complexity (manual download, PATH configuration, multiple platform-specific steps) creates a significant barrier to first use, potentially blocking 30-40% of users from successfully running the tool.

#### Risk Analysis

| Factor | Assessment | Rationale |
|--------|------------|-----------|
| **Likelihood** | High (5) | Windows users face manual installation; PATH configuration is non-obvious for non-technical users; No existing FFmpeg installation for most team members |
| **Impact** | Medium (3) | Blocks core functionality (MKV audio extraction); High support burden; User frustration and abandonment before first successful use |

#### Detailed Mitigation Plan

**1. Comprehensive Platform-Specific Documentation (Week 1 Priority)**

**Linux (Ubuntu/Debian)**:
```bash
sudo apt update && sudo apt install ffmpeg -y
ffmpeg -version
```

**macOS (Homebrew)**:
```bash
brew install ffmpeg
ffmpeg -version
```

**Windows (Manual Installation - PRIORITY)**:
- Download from: https://ffmpeg.org/download.html
- Extract to `C:\ffmpeg\`
- Add `C:\ffmpeg\bin` to System PATH
- Restart terminal and verify: `ffmpeg -version`

**Windows (Chocolatey Alternative)**:
```bash
choco install ffmpeg -y
```

**2. Startup Validation with Helpful Errors**
```python
def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("""
ERROR: FFmpeg not found in PATH.

Installation instructions:
- Linux: sudo apt install ffmpeg
- macOS: brew install ffmpeg
- Windows: https://ffmpeg.org/download.html

See README for detailed Windows PATH setup.
        """)
        sys.exit(1)
```

**3. Windows-Specific Installation Guide**
- Create `docs/INSTALL_WINDOWS_FFMPEG.md` in Week 1
- Include screenshots for PATH configuration
- Link prominently from main README
- Test guide with 2-3 Windows users before release

**4. FFmpeg Version Validation**
- Check minimum version (4.0+) at startup
- Warn if version is outdated with upgrade instructions

**5. Pre-Installation Test Script**
- Provide `check_dependencies.py` for users to verify environment
- Report pass/fail for Python, FFmpeg, API Key with remediation links

**6. Consider Bundled Binaries (v1.1 Enhancement)**
- Research FFmpeg licensing for bundling feasibility
- Decision Point: Week 4 (evaluate installation success rate)

#### Monitoring

| Frequency | Activity | Owner |
|-----------|----------|-------|
| Week 1-2 | Track installation success rate with early adopters | Dev Team |
| Month 1 | Measure: What % of users need installation help? Target: <20% | Tech Lead |
| Month 2 | Reassess bundled binary approach if failure rate >30% | Tech Lead |

#### Success Metrics
- 80%+ users successfully install FFmpeg without support
- Average time-to-first-run <10 minutes (including FFmpeg install)
- <5 GitHub Issues related to FFmpeg installation by Month 2

#### Contingency Plan
- If installation failure rate >40% by Week 4: Fast-track bundled FFmpeg binaries for v0.2 release
- If Windows particularly problematic: Create PowerShell installer script

#### Escalation
- If FFmpeg installation blocks >50% of users: Escalate to Tech Lead for alternative architecture consideration

---

### RISK-003: Large File Handling / Chunking

**Priority**: HIGH (Score: 15)
**Category**: Technical
**Status**: Open
**Owner**: Tech Lead / Developer

*Consolidated from: Primary Draft RISK-004, Technical Risks TECH-003, TECH-007*

#### Description

Processing files >1GB (2+ hour recordings) presents multiple challenges: OpenAI Whisper API has 25MB file size limit requiring chunking, memory constraints during file processing, potential timeouts for long-running API calls, timestamp synchronization across chunks, and complexity of resume logic for interrupted transcriptions. Failure to handle large files blocks a key use case.

#### Risk Analysis

| Factor | Assessment | Rationale |
|--------|------------|-----------|
| **Likelihood** | Medium (3) | 2+ hour recordings are common; API 25MB limit is hard constraint; Chunking adds implementation complexity; Edge cases unpredictable |
| **Impact** | High (4) | Blocks secondary persona's primary use case; User frustration with failed jobs after 30+ minutes; Garbled transcripts if chunking fails |

#### Detailed Mitigation Plan

**1. Automatic File Chunking Implementation (Sprint 4)**

**Chunking Strategy**:
- Target chunk size: 20MB (buffer below 25MB API limit)
- Time-based splitting: 10-minute segments (avoids byte boundary issues)
- Use FFmpeg segment command for streaming (no full memory load):
```bash
ffmpeg -i input.mp3 -f segment -segment_time 600 -c copy chunk_%03d.mp3
```

**2. Chunk Synchronization and Merging**
- Add small overlap (5 seconds) at chunk boundaries
- Deduplicate overlapping text in merge phase
- Track cumulative timestamp offset per chunk
- Validate transcript continuity after merge

**3. Memory-Efficient Processing**
- Stream FFmpeg output to temp files (don't buffer in memory)
- Target: Process 2GB file with <512MB RAM usage
- Use memory profiling during development

**4. Progress Indicators**
- Multi-stage progress bars with `rich` library:
  - Stage 1: "Analyzing file"
  - Stage 2: "Chunking audio" (X of Y chunks)
  - Stage 3: "Transcribing chunks" (X of Y processed)
  - Stage 4: "Merging transcripts"
- Display estimated time remaining

**5. Resume Support for Interrupted Jobs**
- Checkpointing: Save each chunk's transcript immediately
- State tracking in JSON manifest: completed chunks, pending chunks
- CLI flag: `--resume <job-id>` to continue interrupted job

**6. Timeout and Retry Logic**
- 120 second timeout per chunk
- Exponential backoff retry (3 attempts)
- If chunk fails after retries: Log error, skip, continue
- Final report: "11 of 12 chunks successful (chunk 5 failed)"

**7. Testing with Real Large Files (Sprint 4)**
- Test samples: 2-hour podcast, 3-hour conference talk, 4-hour meeting
- Test scenarios: End-to-end, interrupt + resume, network failure, timeout

#### Monitoring

| Frequency | Activity | Owner |
|-----------|----------|-------|
| Sprint 4 | Test with 2-3 hour files, track success rate | Dev Team |
| Month 1 | Track large file (>1GB) success rate | Tech Lead |
| Month 2 | Target: 90%+ success rate for files up to 2GB | Tech Lead |

#### Success Metrics
- 90%+ success rate for files up to 2GB
- Resume functionality works in 95%+ of interrupted scenarios
- SRT output from chunked file plays correctly with video
- <5 GitHub Issues related to large file failures by Month 2

#### Contingency Plan
- If chunking exceeds 2 weeks: Defer large file support to v1.1, document 500MB limit for MVP
- If success rate <80%: Investigate local Whisper model for large file processing

#### Escalation
- If large file handling blocks secondary persona adoption: Escalate to Product Owner for scope decision

---

### RISK-004: Aggressive Timeline

**Priority**: MEDIUM (Score: 12)
**Category**: Schedule
**Status**: Open
**Owner**: Project Manager / Tech Lead

*Consolidated from: Primary Draft RISK-008*

#### Description

The 1-3 month timeline for MVP delivery may be too aggressive given the comprehensive feature set and part-time team availability. Underestimating complexity in FFmpeg integration, API chunking, or error handling could lead to rushed quality, skipped testing, or timeline slippage.

#### Risk Analysis

| Factor | Assessment | Rationale |
|--------|------------|-----------|
| **Likelihood** | Medium (3) | Ambitious feature list for 3 months; Unknowns in FFmpeg integration; Testing often underestimated |
| **Impact** | High (4) | Timeline extension delays ROI; Rushed quality leads to bugs and low adoption; Developer burnout |

#### Mitigation Strategy
1. **Phased MVP Scope**: Phase 1 (Months 1-2) single file + TXT; Phase 2 (Month 3) batch + SRT + large files
2. **2-Week Sprint Reviews**: Assess progress every 2 weeks, adjust timeline at Sprint 2 if velocity <50%
3. **Realistic Estimation**: Include 2-week buffer (total 14 weeks = 3.5 months)
4. **Prefer Working MVP**: Limited features over delayed comprehensive release

#### Contingency Plan
- If timeline extends to 4-5 months: Accept and communicate revised timeline
- If beyond 5 months: Cut scope (defer batch processing or large files to v1.1)

---

### RISK-005: Low Team Adoption

**Priority**: MEDIUM (Score: 12)
**Category**: Business
**Status**: Open
**Owner**: Product Owner / Engineering Team Lead

*Consolidated from: Primary Draft RISK-001*

#### Description

The tool may fail to achieve the 80% team adoption target within 2 months due to poor user experience, installation complexity, or lack of perceived value over existing workflows.

#### Risk Analysis

| Factor | Assessment | Rationale |
|--------|------------|-----------|
| **Likelihood** | Medium (3) | Early internal tools often face adoption challenges; Installation friction creates barriers |
| **Impact** | High (4) | Project ROI depends on team adoption; Below 50% adoption makes payback period unsustainable |

#### Mitigation Strategy
1. **Early Demos**: Week 2 demo to early adopters; Week 4 team-wide demo
2. **Continuous Feedback**: Bi-weekly check-ins, Month 1 survey, Slack channel for support
3. **UX Improvements**: Target 10-minute first-run experience
4. **Documentation Excellence**: Platform-specific guides, quick start, troubleshooting
5. **Team Incentives**: Leadership endorsement, integrate into team workflows

#### Monitoring
- Week 1: Track unique users
- Month 1: Target 40-50% trying tool
- Month 2: Validate 80% regular usage

#### Contingency Plan
- If <50% by Month 1: User interviews to identify blockers
- If <60% by Month 2: Reassess project value

---

### RISK-006: API Key Exposure

**Priority**: MEDIUM (Score: 9)
**Category**: Security
**Status**: Open
**Owner**: Development Team

*Consolidated from: Technical Risks TECH-008, Security Risks SEC-001*

#### Description

OpenAI API key may be accidentally committed to git, logged in debug output, exposed in error messages, or stored insecurely. This could lead to unauthorized API usage, cost overruns, and potential abuse.

#### Risk Analysis

| Factor | Assessment | Rationale |
|--------|------------|-----------|
| **Likelihood** | Medium (3) | Common developer mistake during development or debugging |
| **Impact** | Medium (3) | Financial cost risk, potential API abuse, key revocation disruption |

#### Mitigation Strategy
1. **Environment Variable**: Store API key in `OPENAI_API_KEY` environment variable
2. **Gitignore Enforcement**: Include `.env`, `*.key`, config files in `.gitignore`
3. **Never Log Keys**: Sanitize all logs and error messages, mask key values
4. **Config File Permissions**: If file-based, require 0600 permissions
5. **Pre-commit Hook**: Scan for potential key patterns before commit
6. **Secret Scanning CI**: Use `detect-secrets` or `gitleaks` in pipeline
7. **Documentation**: Clear instructions on secure key management in README

#### Validation Criteria
- [ ] API key loaded from environment variable
- [ ] Config file with insecure permissions rejected
- [ ] All log output verified to not contain API key
- [ ] `.gitignore` includes `.env`, `*.key`, `config.yaml`
- [ ] Security scan in CI pipeline

---

### RISK-007: Command Injection via File Paths

**Priority**: MEDIUM (Score: 9)
**Category**: Security
**Status**: Open
**Owner**: Development Team

*From: Security Risks SEC-002*

#### Description

Maliciously crafted file names or paths could inject shell commands when passed to FFmpeg subprocess calls. Example: A file named `; rm -rf /` could execute unintended commands if paths are not properly sanitized.

#### Risk Analysis

| Factor | Assessment | Rationale |
|--------|------------|-----------|
| **Likelihood** | Low (2) | Internal tool with controlled inputs; team unlikely to intentionally attack |
| **Impact** | Medium (3) | Local system compromise, data loss on user's machine |

#### Mitigation Strategy
1. **Use ffmpeg-python Library**: Handles escaping and quoting automatically
2. **Never Use shell=True**: Use list-based subprocess arguments
3. **Path Validation**: Resolve to absolute paths, reject shell metacharacters
4. **Directory Traversal Prevention**: Reject `../` sequences
5. **Extension Whitelist**: Validate against allowed extensions
6. **Magic Bytes Validation**: Verify file signature, not just extension

#### Validation Criteria
- [ ] Subprocess calls use list arguments, not string concatenation
- [ ] Unit tests with malicious file name inputs
- [ ] Static analysis for `shell=True` usage

---

### RISK-008: API Rate Limits / Batch Processing

**Priority**: MEDIUM (Score: 9)
**Category**: Technical
**Status**: Open
**Owner**: Developer

*Consolidated from: Primary Draft RISK-005, Technical Risks TECH-004*

#### Description

During batch processing (e.g., 20 meeting recordings), the tool may hit OpenAI rate limits causing delays or failed jobs. Users may not understand rate limit errors and abandon the tool.

#### Risk Analysis

| Factor | Assessment | Rationale |
|--------|------------|-----------|
| **Likelihood** | Medium (3) | Batch processing is core feature; Rate limits depend on account tier |
| **Impact** | Medium (3) | Delays in batch completion; Failed jobs require retry; Cost accumulates |

#### Mitigation Strategy
1. **Exponential Backoff**: Use OpenAI SDK retry with 3 attempts, 2x backoff
2. **Configurable Concurrency**: Default 5 concurrent, user override via `--max-concurrent`
3. **Clear Error Messages**: Display rate limit guidance and retry countdown
4. **Cost Tracking**: Estimate cost before batch start, warn if exceeds threshold
5. **Dry-Run Mode**: Show batch plan before execution
6. **Progress Persistence**: Save batch state for resume after failures

#### Validation Criteria
- [ ] Batch of 20 files completes without rate limit errors at default concurrency
- [ ] Cost estimate displayed before batch execution
- [ ] Resume capability after partial batch failure

---

### RISK-009: Audio Format Compatibility

**Priority**: MEDIUM (Score: 9)
**Category**: Technical
**Status**: Open
**Owner**: Developer

*Consolidated from: Primary Draft RISK-009, Technical Risks TECH-005*

#### Description

Real-world files may have unusual codecs, corrupted metadata, or proprietary formats that fail FFmpeg extraction or Whisper API transcription.

#### Risk Analysis

| Factor | Assessment | Rationale |
|--------|------------|-----------|
| **Likelihood** | Medium (3) | Real-world files have format quirks; Team's file sources vary |
| **Impact** | Medium (3) | Specific files fail (not all); Support burden; Manual workarounds |

#### Mitigation Strategy
1. **Format Detection**: Probe files with ffprobe before processing
2. **Conversion Fallback**: Auto-convert unsupported formats to WAV
3. **Clear Error Messages**: Identify format issues specifically with conversion suggestions
4. **Supported Formats Documentation**: List tested/verified formats
5. **Comprehensive Test Suite**: 15+ format variations including edge cases
6. **Graceful Corruption Handling**: Detect corrupted files early

#### Validation Criteria
- [ ] Format detection identifies codec before processing
- [ ] Unsupported codecs trigger automatic WAV conversion
- [ ] Test suite includes diverse format variations

---

### RISK-010: Dependency Vulnerabilities

**Priority**: MEDIUM (Score: 9)
**Category**: Security
**Status**: Open
**Owner**: Development Team

*From: Security Risks SEC-003*

#### Description

Known CVEs in dependencies (`openai`, `ffmpeg-python`, `click`, `rich`, `pydantic`) or transitive dependencies could introduce security vulnerabilities.

#### Risk Analysis

| Factor | Assessment | Rationale |
|--------|------------|-----------|
| **Likelihood** | Medium (3) | Ongoing concern; new CVEs discovered regularly |
| **Impact** | Medium (3) | Varies by vulnerability; could range from disclosure to RCE |

#### Mitigation Strategy
1. **Vulnerability Scanning**: Use `pip-audit` in CI pipeline
2. **Dependabot Alerts**: Enable for automatic notifications
3. **Pin Versions**: All dependencies pinned in `requirements.txt`
4. **Minimal Dependencies**: Only include necessary packages
5. **Regular Updates**: Monthly dependency review
6. **SBOM Generation**: Maintain Software Bill of Materials

#### Validation Criteria
- [ ] CI pipeline includes `pip-audit` (fail on HIGH/CRITICAL)
- [ ] Dependabot enabled on repository
- [ ] Monthly dependency review scheduled

---

### RISK-011: Whisper API Pricing/Policy Changes

**Priority**: LOW (Score: 8)
**Category**: External
**Status**: Open
**Owner**: Product Owner / Project Manager

*Consolidated from: Primary Draft RISK-006, RISK-010, Technical Risks TECH-002*

#### Description

OpenAI may change Whisper API pricing ($0.006/minute currently), increase costs, or modify terms of service. Price increases could make the tool cost-prohibitive ($50/month budget). Policy changes around data privacy could require re-evaluation. API outages would temporarily block functionality.

#### Risk Analysis

| Factor | Assessment | Rationale |
|--------|------------|-----------|
| **Likelihood** | Low (2) | OpenAI APIs are generally stable; Whisper API is mature product |
| **Impact** | High (4) | Pricing >3x makes tool unsustainable; Policy changes may require migration |

#### Mitigation Strategy
1. **Monitor OpenAI Changelog**: Subscribe to API updates, quarterly review
2. **Budget Monitoring**: Track monthly costs, alert at 80% threshold
3. **Abstract API Client**: Create abstraction layer for potential provider switch
4. **Plan Local Whisper Path**: Research whisper.cpp for v2 fallback
5. **Pin SDK Version**: Test before upgrading

#### Contingency Plan
- If pricing increases >2x: Evaluate local Whisper or alternative APIs
- If policy changes violate privacy: Require local Whisper migration

---

### RISK-012: Part-Time Team Availability

**Priority**: LOW (Score: 6)
**Category**: Resource
**Status**: Accepted
**Owner**: Engineering Team Lead

*Consolidated from: Primary Draft RISK-007*

#### Description

Development team (2-5 developers) has competing priorities and may only allocate 20-40% capacity. Part-time allocation could slow velocity, extend timeline, or lead to context-switching inefficiencies.

#### Risk Analysis

| Factor | Assessment | Rationale |
|--------|------------|-----------|
| **Likelihood** | Medium (3) | Internal tools often receive part-time allocation |
| **Impact** | Low (2) | Timeline extension to 4-5 months; Context switching costs |

#### Mitigation Strategy
1. **Ruthless Prioritization**: Focus MVP only, leverage existing libraries
2. **Realistic Sprint Planning**: Assume 20-30% capacity (conservative)
3. **Clear Role Assignments**: Primary developer per module, avoid shared ownership
4. **Knowledge Documentation**: ADRs, README, code comments for continuity
5. **Team Commitment**: Block calendar time, reduce context switching

#### Contingency Plan
- If availability <20%: Pause project or extend timeline to 6 months

---

## Risk Management Process

### Review Cadence

| Review Type | Frequency | Participants | Focus |
|-------------|-----------|--------------|-------|
| Sprint Risk Review | Bi-Weekly | Dev Team | Active risks, new discoveries, status updates |
| Phase Gate Review | Phase Transitions | Team + Stakeholders | Risk retirement, escalations, gate criteria |
| Stakeholder Report | Monthly | PM + Engineering Lead | Top risks, mitigation progress |
| Post-Incident Review | As Needed | Dev Team | Lessons learned, new risk identification |

### Risk Status Definitions

| Status | Definition |
|--------|------------|
| **Open** | Risk identified, mitigation planned but not yet implemented |
| **Mitigating** | Mitigation actions in progress, actively monitoring |
| **Closed** | Risk retired (no longer applicable or successfully mitigated) |
| **Accepted** | Risk acknowledged, team accepts potential impact (no further mitigation) |
| **Realized** | Risk occurred, managing impact |

### Escalation Criteria

Escalate to stakeholders when:
- Risk score increases to Critical (20+)
- New High-priority risk identified (score >= 15)
- Mitigation blocked or ineffective
- Risk realized with significant impact
- External factors change risk profile

### Escalation Path

1. **Developer -> Tech Lead**: Technical risks blocking development
2. **Tech Lead -> Product Owner**: Business/timeline risks requiring scope decisions
3. **Product Owner -> Engineering Team Lead**: High-impact risks threatening project success

### Risk Ownership Matrix

| Category | Primary Owner | Escalation |
|----------|---------------|------------|
| Business | Product Owner | Engineering Team Lead |
| Technical | Tech Lead | Engineering Manager |
| Security | Development Team | Security Architect |
| Schedule | Project Manager | Product Owner |
| Resource | Engineering Team Lead | Executive Sponsor |
| External | Project Manager | Product Owner |

---

## Risk Score Matrix

| Impact | Low (2) | Medium (3) | High (4) | Show Stopper (5) |
|--------|---------|------------|----------|------------------|
| **High (5)** | 10 | 15 | **20** | 25 |
| **Medium (3)** | 6 | 9 | 12 | 15 |
| **Low (2)** | 4 | 6 | 8 | 10 |

**Priority Thresholds**:
- **HIGH** (15-25): Detailed mitigation plan, weekly monitoring, immediate action
- **MEDIUM** (9-14): Standard mitigation, bi-weekly monitoring
- **LOW** (< 9): Accept or monitor, monthly review

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| **Total Risks** | 12 |
| **HIGH Priority** | 3 (25%) |
| **MEDIUM Priority** | 7 (58%) |
| **LOW Priority** | 2 (17%) |
| **Show Stoppers** | 0 |
| **Open** | 9 |
| **Mitigating** | 2 |
| **Accepted** | 1 |

### Risks by Category

| Category | Count | Highest Score |
|----------|-------|---------------|
| Technical | 4 | 15 (RISK-002, RISK-003) |
| Business | 2 | 20 (RISK-001) |
| Security | 3 | 9 |
| Schedule | 1 | 12 |
| Resource | 1 | 6 |
| External | 1 | 8 |

---

## Related Documents

- Vision Document: `/home/manitcor/dev/tnf/.aiwg/requirements/vision-document.md`
- Project Intake: `/home/manitcor/dev/tnf/.aiwg/intake/project-intake.md`
- Technical Risks (Source): `/home/manitcor/dev/tnf/.aiwg/working/risks/risk-list/drafts/technical-risks.md`
- Security Risks (Source): `/home/manitcor/dev/tnf/.aiwg/working/risks/risk-list/drafts/security-risks.md`

---

## Sign-Off

**Required Approvals:**

| Role | Status | Name | Date |
|------|--------|------|------|
| Project Manager | APPROVED | - | 2025-12-04 |
| Architecture Designer | APPROVED | - | 2025-12-04 |
| Security Architect | APPROVED | - | 2025-12-04 |
| Tech Lead | PENDING | - | - |
| Product Owner | PENDING | - | - |

**Conditions**: None

**Outstanding Concerns**: None

---

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-04 | Documentation Synthesizer | Consolidated from 3 source documents; Merged duplicates; Prioritized by severity; Added detailed mitigation plans for top 3 risks |
| 0.1 | 2025-12-04 | Project Manager | Initial primary draft - 10 risks |

---

**Document Status**: BASELINED
**Next Review**: Phase Gate (Inception -> Elaboration)
