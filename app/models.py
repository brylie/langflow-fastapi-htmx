from pydantic import BaseModel


class RagCitation(BaseModel):
    source: str
    content: str

    def __str__(self):
        # Truncate content for display
        return f"{self.source}: {self.content[:50]}..."
