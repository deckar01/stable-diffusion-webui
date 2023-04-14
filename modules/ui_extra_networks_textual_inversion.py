import json
import os

from modules import ui_extra_networks, sd_hijack, shared
from modules.textual_inversion.textual_inversion import Embedding


class ExtraNetworksPageTextualInversion(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Textual Inversion')
        self.allow_negative_prompt = True

    def refresh(self):
        sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)

    def list_items(self):
        embeddings = [emb for emb in sd_hijack.model_hijack.embedding_db.word_embeddings.values()]
        if len(embeddings) == 0: # maybe not loaded yet, so lets just look them up
            for root, _dirs, fns in os.walk(shared.opts.embeddings_dir):
                for fn in fns:
                    embedding = Embedding(0, fn)
                    embedding.filename = os.path.join(root, fn)
                    embeddings.append(embedding)
        for embedding in embeddings:
            path, ext = os.path.splitext(embedding.filename)
            yield {
                "name": embedding.name,
                "filename": embedding.filename,
                "preview": self.find_preview(path),
                "description": self.find_description(path),
                "search_term": self.search_terms_from_path(embedding.filename),
                "prompt": json.dumps(embedding.name),
                "local_preview": f"{path}.preview.{shared.opts.samples_format}",
            }

    def allowed_directories_for_previews(self):
        return list(sd_hijack.model_hijack.embedding_db.embedding_dirs)
