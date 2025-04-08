import numpy

class Database:
    def __init__(self):
        self.face_db = {}
        self.initialized = False
        
    def populate_database(self, jsonData):
        for name, embeddings in jsonData.items():
            normalized_embeddings = []
            for embedding in embeddings:
                embedding_array = numpy.array(embedding)
                normalized_embedding = self.normalize_vector(embedding_array)
                normalized_embeddings.append(normalized_embedding)
            self.face_db[name] = normalized_embeddings
        self.initialized = True
    
    def normalize_vector(self, vector):
        return vector / numpy.linalg.norm(vector)

    def cosine_similarity_search(self, query_embedding):
        query_embedding = numpy.array(query_embedding)
        query_embedding = self.normalize_vector(query_embedding)
        best_match = None
        best_match_score = -1
        for name, embeddings in self.face_db.items():
            for embedding in embeddings:
                score = numpy.dot(embedding, query_embedding)   # Cosine similarity
                if score > best_match_score:
                    best_match_score = score
                    best_match = name
        return {"name": best_match, "score": best_match_score}