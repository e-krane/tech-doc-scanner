I want to build a document conversion agent powered by docling. It should convert technical documents containing text, equations, images, tables, etc. to markdown files and appended figures.

It should:
1. Be easy to use.
2. Make quality conversions for later use in a rag pipeline.
3. Be performant, using local GPU resources.
4. Optimize the flags passed to docling based on each document.
5. Use smart chunking strategies from docling.