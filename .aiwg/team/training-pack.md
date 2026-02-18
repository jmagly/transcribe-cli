# Training Pack

Owner: Engineering Manager  
Version: 1.0  
Status: DRAFT  
Last Updated: 2026-02-17

## Audience
- Developers onboarding to `transcribe-cli`
- Support engineers handling production issues

## Training Modules
1. Environment setup and dependency validation
2. CLI workflows: transcribe/extract/batch/config/setup-check
3. Test execution and coverage gate interpretation
4. Release, rollback, and support handoff workflow

## Hands-on Checklist
- Run `.venv/bin/pytest -q`
- Validate `transcribe --help`
- Review gate records in `.aiwg/gates/`
- Review deployment and rollback plans
