# Data Classification Document

**Project**: Audio Transcription CLI Tool
**Version**: 1.0
**Date**: 2025-12-04
**Author**: Security Architect
**Status**: DRAFT

---

## Executive Summary

This document establishes the data classification framework for the Audio Transcription CLI Tool. As a local CLI application processing user audio files and interfacing with OpenAI's Whisper API, the tool handles a limited set of data types with varying sensitivity levels. The primary security concern is protecting the OpenAI API key (Confidential), while other data types are classified as Internal with standard handling requirements.

---

## 1. Data Types Inventory

| Data Type | Description | Classification | Handling Requirements |
|-----------|-------------|----------------|----------------------|
| Audio Files | User-provided audio/video files (MP3, AAC, FLAC, WAV, M4A, MKV) | Internal | Local processing only, not stored by tool, user-controlled |
| Transcripts | Generated text output from Whisper API | Internal | Saved to user's local filesystem, user-controlled retention |
| API Key | OpenAI API credential (OPENAI_API_KEY) | **Confidential** | Environment variable, never logged, never committed, masked in errors |
| Config Settings | User preferences (output directory, formats, language settings) | Internal | Local config file (~/.transcriberc or .env), no sensitive data |
| Temporary Files | Extracted audio tracks, audio chunks for API submission | Internal | Secure temp directory (tempfile module), automatic cleanup on completion |
| Log Output | CLI operation logs (progress, status, errors) | Internal | No sensitive data logged, file paths only (not content) |
| Metadata | Audio file metadata (duration, format, sample rate, size) | Internal | Extracted for processing decisions, not persisted |
| API Responses | Whisper API transcription responses | Internal | Parsed and written to output files, not cached |

---

## 2. Classification Definitions

### Public

- **Definition**: Information intended for public disclosure with no restrictions
- **Access Control**: None required
- **Encryption**: Not required
- **Examples**: Open-source code, public documentation, marketing materials
- **Applicability to This Project**: None (no public data types)

### Internal

- **Definition**: Information for organizational use only, not intended for external sharing
- **Access Control**: Limited to authorized personnel/systems
- **Encryption**: Recommended for transmission, optional at rest
- **Examples**: Audio files, transcripts, configuration settings, logs
- **Applicability to This Project**: Most data types fall into this category

### Confidential

- **Definition**: Sensitive information requiring restricted access and protection
- **Access Control**: Need-to-know basis, role-based access
- **Encryption**: Required for transmission, recommended at rest
- **Audit**: Logging of access events recommended
- **Examples**: API credentials, authentication tokens, encryption keys
- **Applicability to This Project**: OpenAI API Key

### Restricted

- **Definition**: Highest sensitivity classification requiring strict controls
- **Access Control**: Strict need-to-know, explicit authorization required
- **Encryption**: Mandatory at rest and in transit
- **Audit**: Comprehensive logging required
- **Examples**: PII, PHI, financial data, trade secrets
- **Applicability to This Project**: None (no restricted data types identified)

---

## 3. Security Requirements by Classification

### Confidential Data (API Key)

| Control Area | Requirement | Implementation |
|--------------|-------------|----------------|
| **Storage** | Secure storage, never in source code | Environment variable (OPENAI_API_KEY) or config file with 0600 permissions |
| **Transmission** | Encrypted channel only | HTTPS enforced by OpenAI SDK (TLS 1.2+) |
| **Logging** | Never log or display | Mask in error messages, exclude from verbose output |
| **Access** | Limited to authorized users | OS-level file permissions, user environment |
| **Disposal** | Clear from memory after use | Python garbage collection, avoid unnecessary copies |
| **Version Control** | Never commit | .gitignore entry for .env, secrets scanning in CI |
| **Rotation** | Periodic rotation recommended | Document rotation procedure in README |

**Implementation Details**:

```python
# Secure API key handling
import os
from pydantic import SecretStr

class Config:
    api_key: SecretStr = SecretStr(os.environ.get("OPENAI_API_KEY", ""))

    def get_api_key(self) -> str:
        """Returns API key value. Never log this."""
        return self.api_key.get_secret_value()
```

**Error Message Masking**:

```python
# BAD: Exposes API key
raise Exception(f"API call failed with key: {api_key}")

# GOOD: Masks API key
raise Exception("API call failed. Check OPENAI_API_KEY environment variable.")
```

### Internal Data (Audio Files, Transcripts, Temp Files, Logs)

| Control Area | Requirement | Implementation |
|--------------|-------------|----------------|
| **Storage** | User's local filesystem | Default output directory, configurable via settings |
| **Transmission** | Local processing, encrypted API calls | FFmpeg local processing, HTTPS for Whisper API |
| **Logging** | File paths acceptable, not content | Log file paths, sizes, formats; never log transcript content |
| **Access** | User's local permissions | OS file permissions apply |
| **Disposal** | User responsibility for audio/transcripts, auto-cleanup for temp | tempfile.TemporaryDirectory with automatic cleanup |
| **Version Control** | User decision | Recommend excluding transcripts from VCS if sensitive |

**Temporary File Security**:

```python
import tempfile
import atexit
import shutil

# Secure temporary directory
temp_dir = tempfile.mkdtemp(prefix="transcribe_")

# Register cleanup on exit
atexit.register(lambda: shutil.rmtree(temp_dir, ignore_errors=True))

# Cleanup on completion
def cleanup_temp_files():
    """Remove temporary audio chunks and extracted files."""
    shutil.rmtree(temp_dir, ignore_errors=True)
```

---

## 4. Data Flow Diagram

### Text-Based Data Flow

```
+------------------+
| User Audio File  |
| (Internal)       |
| .mp3/.mkv/.flac  |
+--------+---------+
         |
         v
+------------------+
| CLI Tool         |
| [Local Machine]  |
+--------+---------+
         |
         | FFmpeg Extract (if MKV)
         v
+------------------+
| Temp Audio File  |
| (Internal)       |
| Auto-cleanup     |
+--------+---------+
         |
         | + API Key (Confidential)
         | HTTPS Transport
         v
+------------------+
| OpenAI Whisper   |
| API              |
| [External]       |
+--------+---------+
         |
         | Transcript Response
         | HTTPS Transport
         v
+------------------+
| Output File      |
| (Internal)       |
| .txt/.srt/.vtt   |
+--------+---------+
         |
         v
+------------------+
| Temp File        |
| Cleanup          |
+------------------+
```

### Data Flow Summary

| Step | Data | Classification | Protection |
|------|------|----------------|------------|
| 1. Input | Audio/Video File | Internal | Local filesystem |
| 2. Extract | Audio Track | Internal | Temp directory, auto-cleanup |
| 3. Chunk | Audio Segments | Internal | Temp files, <25MB each |
| 4. Transmit | Audio + API Key | Confidential (key) | HTTPS/TLS 1.2+ |
| 5. Process | Whisper API | External Service | OpenAI security controls |
| 6. Receive | Transcript | Internal | HTTPS response |
| 7. Output | Transcript File | Internal | Local filesystem |
| 8. Cleanup | Temp Files | Internal | Automatic deletion |

### Trust Boundaries

```
+----------------------------------------------------------+
|                    User's Local Machine                   |
|                    (Trust Zone: Internal)                 |
|                                                          |
|  +-------------+     +-------------+     +-------------+ |
|  | Audio Files |---->| CLI Tool    |---->| Transcripts | |
|  +-------------+     +------+------+     +-------------+ |
|                            |                             |
+----------------------------|-----------------------------+
                             |
                    [Trust Boundary: Network/TLS]
                             |
                             v
+----------------------------------------------------------+
|                    OpenAI Cloud                           |
|                    (Trust Zone: External/Vendor)          |
|                                                          |
|  +-------------+     +-------------+     +-------------+ |
|  | API Gateway |---->| Whisper API |---->| Response    | |
|  +-------------+     +-------------+     +-------------+ |
|                                                          |
+----------------------------------------------------------+
```

---

## 5. Compliance Mapping

### Regulatory Assessment

| Framework | Applicability | Rationale |
|-----------|---------------|-----------|
| **GDPR** | Not Applicable | No EU personal data stored or processed by the tool. Audio files remain on user's local machine. OpenAI handles their own GDPR compliance. |
| **CCPA** | Not Applicable | No California consumer data collected or stored. Local CLI tool. |
| **HIPAA** | Not Applicable | No Protected Health Information (PHI) processed. Users are responsible for content in their audio files. |
| **PCI-DSS** | Not Applicable | No payment card data processed or stored. |
| **SOC 2** | Not Applicable | Not a SaaS service. CLI tool runs locally. |
| **ISO 27001** | Informational | Good security practices align with ISO 27001 controls but certification not required. |

### User Responsibility Notice

**Important**: The Audio Transcription CLI Tool does not inspect or classify the content of audio files. Users are solely responsible for:

1. Ensuring audio content complies with applicable regulations (HIPAA, GDPR, etc.)
2. Not processing sensitive regulated data without appropriate controls
3. Complying with OpenAI's usage policies and terms of service
4. Managing transcript retention and deletion per their organization's policies

### OpenAI Data Processing

Per OpenAI's current data usage policies:

- Whisper API does **not** retain audio data after processing
- Audio is processed transiently and not used for model training (for API customers)
- Users should review [OpenAI's Privacy Policy](https://openai.com/policies/privacy-policy) for current terms

---

## 6. Security Controls Summary

### Control Implementation Matrix

| Control | Implementation | Validation Method | Priority |
|---------|----------------|-------------------|----------|
| **API Key Protection** | Environment variable (OPENAI_API_KEY), .gitignore for .env | CI pre-commit hook for secrets detection (detect-secrets, gitleaks) | Critical |
| **Input Validation** | Path sanitization (os.path.realpath), format detection (python-magic), size warnings | Unit tests for path traversal, format edge cases | High |
| **Temp File Security** | tempfile.mkdtemp(), atexit cleanup handler, explicit cleanup on success/failure | Integration tests verify no temp file leakage | High |
| **Dependency Security** | pip-audit in CI, Dependabot alerts, requirements.txt version pinning | Automated weekly scans, PR checks | High |
| **Transport Security** | HTTPS enforced by OpenAI SDK (TLS 1.2+) | Integration test verifies HTTPS endpoint | High |
| **Logging Hygiene** | No API keys or transcript content in logs, structured JSON logging | Code review checklist, log audit | Medium |
| **Error Handling** | Mask sensitive data in exceptions, user-friendly error messages | Unit tests for error message content | Medium |
| **File Permissions** | Config files created with 0600 permissions, user-owned temp directories | Manual verification, integration tests | Medium |

### Pre-Release Security Checklist

- [ ] No hardcoded API keys in source code (verified by secrets scanner)
- [ ] .gitignore includes .env, *.key, credentials files
- [ ] pip-audit shows no high/critical vulnerabilities
- [ ] All API calls use HTTPS (verified in code review)
- [ ] Error messages do not expose API keys or sensitive paths
- [ ] Temp file cleanup implemented and tested
- [ ] Input validation prevents directory traversal attacks
- [ ] README includes security best practices section

---

## 7. Data Retention and Disposal

### Retention Periods

| Data Type | Retention | Disposal Method |
|-----------|-----------|-----------------|
| Audio Files | User-controlled | User's responsibility (tool does not delete source files by default) |
| Transcripts | User-controlled | User's responsibility |
| Temp Files | Duration of processing | Automatic deletion via tempfile cleanup |
| Logs | Session-based (stdout/stderr) | No persistent storage unless user redirects |
| API Key | Permanent (user-managed) | User rotates/revokes via OpenAI dashboard |
| Config | Permanent (until user deletes) | User deletes config file manually |

### Secure Disposal Procedures

**Temporary Files**:

```python
import shutil
import os

def secure_cleanup(temp_dir: str) -> None:
    """Securely remove temporary directory and contents."""
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        # Log warning but don't expose path details
        logging.warning("Temp cleanup incomplete. Manual cleanup may be required.")
```

**User Guidance for Sensitive Audio**:

Users processing sensitive audio should:

1. Use encrypted filesystem for output directory
2. Implement organization's data retention policy for transcripts
3. Consider `--delete-source` flag (future feature) for one-time transcription

---

## 8. Third-Party Data Handling

### OpenAI Whisper API

| Aspect | Details |
|--------|---------|
| **Data Transmitted** | Audio file content (chunked if >25MB) |
| **Data Classification** | Internal (content), Confidential (API key in header) |
| **Transport Security** | TLS 1.2+ (HTTPS enforced by SDK) |
| **Data Retention by Vendor** | Zero-day retention (audio not stored after processing) |
| **Data Location** | OpenAI cloud infrastructure (US-based) |
| **Sub-processors** | See [OpenAI Sub-processors](https://openai.com/policies/service-terms) |

### Vendor Security Assessment

| Control | OpenAI Status | Evidence |
|---------|---------------|----------|
| SOC 2 Type II | Compliant | OpenAI Trust Portal |
| Encryption in Transit | TLS 1.2+ | API endpoint verification |
| Data Processing Agreement | Available | OpenAI DPA for enterprise |
| Incident Response | Documented | OpenAI security page |

---

## 9. Appendix: Classification Decision Tree

```
Is the data an API credential, token, or secret?
    YES --> CONFIDENTIAL
    NO  --> Continue

Does the data contain PII, PHI, or financial information?
    YES --> RESTRICTED (consult legal/compliance)
    NO  --> Continue

Is the data intended for public disclosure?
    YES --> PUBLIC
    NO  --> INTERNAL
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-04 | Security Architect | Initial data classification document |

---

## Approvals

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Security Architect | | | |
| Project Lead | | | |
| Technical Architect | | | |
