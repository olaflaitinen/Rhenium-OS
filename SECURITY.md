# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability within Rhenium OS, please send an email to the maintainers. All security vulnerabilities will be promptly addressed.

**Please do not report security vulnerabilities through public GitHub issues.**

### What to include

- Type of vulnerability
- Full paths of source file(s) related to the vulnerability
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Resolution**: Depends on severity and complexity

## Security Considerations for Medical AI

> **IMPORTANT**: This software is for research purposes only and is not approved for clinical use.

### Data Privacy

- All patient data must be de-identified before processing
- Follow DICOM PS3.15 de-identification guidelines
- Never commit real patient data to the repository
- Use synthetic data for testing and development

### AI Safety

- All AI-generated findings require human verification
- Generated images are automatically marked with disclosure metadata
- Model outputs should not be used as the sole basis for clinical decisions
