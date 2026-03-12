import faiss
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from numpy.linalg import norm
import ollama
import os
import transformers

transformers.utils.logging.set_verbosity_error()

NOME_MODELO = "openai/clip-vit-base-patch32"
MODELO_OLLAMA = "gemma3"

documentos = [
    "Python é uma linguagem de programação de alto nível",
    "Machine learning é um subcampo da inteligência artificial",
    "RAG combina recuperação de informação com geração de texto",
    "Embeddings são representações vetoriais de texto",
    "Transformers são modelos de deep learning para NLP"
]

class RAG:
    def __init__(self):
        #self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        self.model = CLIPModel.from_pretrained(NOME_MODELO).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(NOME_MODELO)
        self.embeddings = None
        self.index = None
        self.documentos_referencia = []

    def gerarEmbeddingTexto(self, texto):
        inputs = self.processor(text=[texto], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        
        vetor = features.cpu().numpy()[0]
        vetor = vetor.astype('float32') #float32 é uma exigência do FAISS
        return vetor / norm(vetor)

    def gerarEmbeddings(self, documentos):
        self.documentos_referencia = documentos
        # Processando em lote (batch) em vez de um por um
        inputs = self.processor(text=documentos, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        
        # Normalização vetorial (L2) para similaridade de cosseno
        embeddings = features.cpu().numpy().astype('float32')
        faiss.normalize_L2(embeddings) 
        self.embeddings = embeddings

    def armazenarEmbeddingBancoVetorial(self):
        dimensao = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimensao)
        self.index.add(self.embeddings)

    def indexacao(self, documentos):
        self.gerarEmbeddings(documentos)
        self.armazenarEmbeddingBancoVetorial()

    def recuperacao(self, pergunta, k=1):
        # 1. Gerar embedding da pergunta
        embedding_pergunta = self.gerarEmbeddingTexto(pergunta).reshape(1, -1)

        # 2. Pesquisar no FAISS
        # I contém os índices dos documentos mais próximos
        score, indice = self.index.search(embedding_pergunta, k) #score = distância
        
        self.score = score
        self.indice = indice

        print(score, indice)
        
        # 3. Retornar os documentos correspondentes da nossa lista de referência
        resultados = [self.documentos_referencia[i] for i in indice[0]]

        return resultados

    # Integração com o Ollama
    def geracao(self, pergunta, contexto, system_prompt=None):  
        prompt = pergunta

        system_prompt = f"""
            Você é um assistente útil. Use o contexto abaixo para responder à pergunta.
            
            Contexto:
            {contexto}
            
            Pergunta:
            {pergunta}
        """

        messages = []
    
        if system_prompt:
            messages.append({ 'role': 'system','content': system_prompt })
        
        messages.append({ 'role': 'user', 'content': prompt })
        
        try:
            response = ollama.chat(model=MODELO_OLLAMA, messages=messages, options={
                'temperature': 0.7,
                'num_predict': 1000
            })

            return response['message']['content']

        except Exception as e:
            raise Exception(f"Erro na requisição Ollama: {e}")

# --- Teste do Fluxo ---
rag = RAG()
rag.indexacao(documentos)

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU") # Retorna True se a GPU estiver pronta para uso
#print(torch.cuda.get_device_name(0))

pergunta = "O que são vetores de texto?"
resultado = rag.recuperacao(pergunta, k=5)

print(f"Pergunta: {pergunta}")
print(f"Documento recuperado: {resultado[0]}")

print(f"Levando pergunta ao Ollama")
resposta = rag.geracao(pergunta, resultado[0])

print(f"Resposta do Ollama: {resposta}")