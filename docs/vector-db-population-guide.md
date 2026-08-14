# Vector DB Population Guide

## Goal

Populate a vector database with synthetic business policy documents so the GenAI prototype can answer workplace policy questions with grounded citations, risk flags, and practical next steps.

## Data Source

Use the Markdown files in:

`data/policies/`

Each document includes frontmatter metadata and numbered Markdown sections. The frontmatter is useful for filtering and citation display. The section headings are useful chunk boundaries.

## Recommended Ingestion Flow

1. Read each Markdown file.
2. Parse frontmatter metadata.
3. Split the body by `##` section headings.
4. Create one vector record per section.
5. Store the section text plus metadata.
6. Embed each section using your selected embedding model.
7. Save vectors in your chosen vector database.

## Recommended Record Shape

```json
{
  "id": "POL-001::3-home-office-equipment",
  "text": "## 3. Home Office Equipment...",
  "metadata": {
    "document_id": "POL-001",
    "title": "Employee Expense Reimbursement Policy",
    "department": "Finance",
    "version": "1.2",
    "effective_date": "2026-01-15",
    "section": "3. Home Office Equipment",
    "source_file": "employee-expense-reimbursement-policy.md",
    "classification": "Synthetic demo data"
  }
}
```

## Retrieval Settings

Start simple:

- Retrieve top 4 to 6 chunks.
- Require answers to cite document title and section.
- If retrieved chunks do not directly answer the question, return an insufficient-context response.
- Add a risk category such as `Finance`, `Legal`, `Security`, `Privacy`, `HR`, or `Low`.

## Prompt Contract

Use a prompt contract like this:

```text
You are a business policy assistant.
Answer only from the provided policy excerpts.
If the answer is not supported by the excerpts, say that the policy does not provide enough information.
Include citations using document title and section.
Flag any HR, Finance, Legal, Security, or Privacy risk.
End with one practical next step.
```

## Portfolio Framing

In the GitHub README, describe the vector database as the grounding layer for a business-safe GenAI assistant. Recruiters should be able to see that the project covers:

- Business use case framing
- Retrieval-augmented generation
- Grounded answers with citations
- Responsible AI controls
- Risk escalation
- Low-cost cloud deployment path

