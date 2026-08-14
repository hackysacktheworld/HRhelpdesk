# Demo Policy Document Set

This folder contains synthetic business policy documents for a recruiter-facing GenAI prototype. The documents are designed for retrieval-augmented generation, vector database demos, citations, and responsible AI testing.

## Suggested Chunking

- Split by Markdown heading, especially `##` sections.
- Preserve `document_id`, `title`, `department`, `version`, and `effective_date` as metadata on every chunk.
- Use the section heading as a citation anchor.
- Keep chunks around 300 to 800 tokens when possible.

## Suggested Metadata Fields

- `document_id`
- `title`
- `department`
- `version`
- `effective_date`
- `section`
- `source_file`
- `classification`

## Good Demo Questions

- Can I get reimbursed for a home office monitor?
- Can employees paste customer records into an AI tool?
- When does a vendor need Security review?
- What should support do if a customer reports unauthorized account access?
- What approval is needed for a 12,000 USD software purchase?
- What should I do if my company laptop is stolen?
- Can support promise a customer a fix date during an outage?
- When are competitive quotes required?
- Who can approve an exception to the AI policy?
- What happens if an employee submits an expense after 60 days?

## Responsible AI Notes

These documents are synthetic and should be labeled as demo data in public repositories. The prototype should instruct the model to answer only from retrieved context, cite the relevant document and section, and escalate when the source material does not contain enough information.

