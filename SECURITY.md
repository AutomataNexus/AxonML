# Security Policy

## Scope

This security policy covers all software and firmware maintained under the AutomataNexus organization, including but not limited to NexusEdge controller firmware, NexusBMS, NexusConnect, and associated tools and libraries. Given that these systems operate in commercial building automation environments controlling HVAC and other critical infrastructure, we take security reports seriously and prioritize rapid response.

## Supported Versions

Only the latest release branch of each project receives security patches. If you are running an older version, please upgrade before reporting unless you believe the vulnerability also affects the current release.

| Branch       | Supported          |
| ------------ | ------------------ |
| Latest       | :white_check_mark: |
| Previous     | :warning: Critical fixes only |
| All others   | :x:                |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, please report vulnerabilities privately using one of the following methods:

1. **GitHub Private Vulnerability Reporting** — Use the "Report a vulnerability" button under the Security tab of the affected repository. This is the preferred method.
2. **Email** — Send a detailed report to **security@automatanexus.com**. If you'd like to encrypt your message, our PGP key is available upon request.

### What to Include

- Affected project and version (or commit hash)
- Description of the vulnerability and its potential impact
- Steps to reproduce or a proof of concept
- Any suggested remediation if you have one

### What to Expect

- **Acknowledgment** within 48 hours confirming we received your report.
- **Initial assessment** within 7 days with a severity classification and our planned response.
- **Status updates** at least every 14 days until the issue is resolved or a decision is made.
- **Credit** in the release notes and/or advisory unless you prefer to remain anonymous.

If a reported vulnerability is accepted, we will work on a fix, coordinate disclosure, and issue a security advisory through GitHub. If declined, we will explain our reasoning.

## Disclosure Policy

We follow coordinated disclosure. We ask reporters to give us a reasonable window (typically 90 days) to address the vulnerability before any public disclosure. We will work with you on timing and are happy to coordinate joint announcements.

## Security Considerations for Deployments

If you are deploying AutomataNexus software in production environments:

- Always run the latest release and subscribe to repository notifications for security advisories.
- NexusEdge controllers should operate on isolated network segments, not exposed directly to the public internet.
- Use TLS for all API communications and rotate credentials regularly.
- Review the hardening guide in each project's documentation before deploying to production.

## Contact

For general security questions that are not vulnerability reports, you can reach us at **security@automatanexus.com** or open a discussion on the relevant repository.
