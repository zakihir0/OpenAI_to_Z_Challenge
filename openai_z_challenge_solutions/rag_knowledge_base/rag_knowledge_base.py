# RAG System for Archaeological Knowledge Base - OpenAI to Z Challenge
# Implementation based on best practices for RAG with LLM and vector stores

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import openai
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Try to import optional dependencies
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

@dataclass
class ArchaeologicalSite:
    """Data structure for archaeological site information"""
    name: str
    coordinates: tuple
    culture: str
    time_period: str
    site_type: str
    characteristics: List[str]
    discovery_method: str
    significance: str
    references: List[str]

@dataclass
class KnowledgeDocument:
    """Data structure for knowledge base documents"""
    id: str
    title: str
    content: str
    document_type: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class EmbeddingGenerator:
    """Generate embeddings using OpenAI's embedding models"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-large"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.embedding_cache = {}
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float"
            )
            embedding = response.data[0].embedding
            self.embedding_cache[text] = embedding
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 3072  # text-embedding-3-large dimension
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        embeddings = []
        for text in texts:
            embedding = self.generate_embedding(text)
            embeddings.append(embedding)
        return embeddings

class VectorStore:
    """Abstract base class for vector storage implementations"""
    
    def add_documents(self, documents: List[KnowledgeDocument]):
        raise NotImplementedError
    
    def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[tuple]:
        raise NotImplementedError
    
    def get_document(self, doc_id: str) -> Optional[KnowledgeDocument]:
        raise NotImplementedError

class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store implementation"""
    
    def __init__(self):
        self.documents: Dict[str, KnowledgeDocument] = {}
        self.embeddings: Dict[str, List[float]] = {}
    
    def add_documents(self, documents: List[KnowledgeDocument]):
        """Add documents to the vector store"""
        for doc in documents:
            self.documents[doc.id] = doc
            if doc.embedding:
                self.embeddings[doc.id] = doc.embedding
    
    def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[tuple]:
        """Find most similar documents using cosine similarity"""
        if not self.embeddings:
            return []
        
        # Calculate similarities
        similarities = []
        query_embedding = np.array(query_embedding).reshape(1, -1)
        
        for doc_id, doc_embedding in self.embeddings.items():
            doc_embedding = np.array(doc_embedding).reshape(1, -1)
            similarity = cosine_similarity(query_embedding, doc_embedding)[0][0]
            similarities.append((doc_id, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def get_document(self, doc_id: str) -> Optional[KnowledgeDocument]:
        """Retrieve document by ID"""
        return self.documents.get(doc_id)

class ChromaVectorStore(VectorStore):
    """ChromaDB vector store implementation"""
    
    def __init__(self, collection_name: str = "archaeological_knowledge"):
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is not installed. Install with: pip install chromadb")
        
        self.client = chromadb.Client()
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.documents: Dict[str, KnowledgeDocument] = {}
    
    def add_documents(self, documents: List[KnowledgeDocument]):
        """Add documents to ChromaDB collection"""
        ids = []
        embeddings = []
        metadatas = []
        texts = []
        
        for doc in documents:
            self.documents[doc.id] = doc
            ids.append(doc.id)
            embeddings.append(doc.embedding)
            metadatas.append(doc.metadata)
            texts.append(doc.content)
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts
        )
    
    def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[tuple]:
        """Find similar documents using ChromaDB"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        # Convert ChromaDB results to our format
        similarities = []
        if results['ids'] and results['distances']:
            for doc_id, distance in zip(results['ids'][0], results['distances'][0]):
                similarity = 1 - distance  # Convert distance to similarity
                similarities.append((doc_id, similarity))
        
        return similarities
    
    def get_document(self, doc_id: str) -> Optional[KnowledgeDocument]:
        """Retrieve document by ID"""
        return self.documents.get(doc_id)

class ArchaeologicalKnowledgeBase:
    """Comprehensive knowledge base for Amazon archaeological sites"""
    
    def __init__(self):
        self.sites = []
        self.documents = []
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Initialize with known archaeological sites and research data"""
        
        # Major archaeological sites in the Amazon basin
        known_sites = [
            ArchaeologicalSite(
                name="Monte Alegre",
                coordinates=(-2.01, -54.07),
                culture="Early Paleoindian",
                time_period="11,200-10,000 BP",
                site_type="Rock Art Complex",
                characteristics=["Cave paintings", "Rock shelters", "Early human occupation"],
                discovery_method="Archaeological excavation",
                significance="Some of the earliest human occupation evidence in the Amazon",
                references=["Roosevelt et al. 1996", "Roosevelt 2002"]
            ),
            ArchaeologicalSite(
                name="Marajoara Complex",
                coordinates=(-1.0, -50.0),
                culture="Marajoara",
                time_period="400-1350 CE",
                site_type="Complex Society Settlement",
                characteristics=["Large earthwork mounds", "Elaborate ceramics", "Social stratification"],
                discovery_method="Systematic survey",
                significance="Evidence of complex pre-Columbian societies in Amazonia",
                references=["Schaan 2004", "Roosevelt 1991"]
            ),
            ArchaeologicalSite(
                name="Acre Geoglyphs",
                coordinates=(-9.97, -67.81),
                culture="Pre-Columbian Amazonian",
                time_period="200-1450 CE",
                site_type="Earthwork Complex",
                characteristics=["Geometric earthworks", "Deforestation patterns", "Ritual spaces"],
                discovery_method="Remote sensing and deforestation analysis",
                significance="Large-scale landscape modification by pre-Columbian peoples",
                references=["Pärssinen et al. 2009", "Watling et al. 2017"]
            ),
            ArchaeologicalSite(
                name="Llanos de Mojos",
                coordinates=(-15.0, -65.0),
                culture="Pre-Columbian Hydraulic",
                time_period="300-1400 CE",
                site_type="Agricultural Complex",
                characteristics=["Raised fields", "Drainage systems", "Settlement mounds"],
                discovery_method="Aerial photography and excavation",
                significance="Sophisticated pre-Columbian agricultural systems",
                references=["Erickson 2000", "Walker 2004"]
            ),
            ArchaeologicalSite(
                name="Santarém Culture Sites",
                coordinates=(-2.44, -54.71),
                culture="Santarém/Tapajós",
                time_period="1000-1500 CE",
                site_type="Urban Settlement",
                characteristics=["Elaborate ceramics", "Trade networks", "Urban planning"],
                discovery_method="Archaeological excavation",
                significance="Evidence of complex urban societies in central Amazon",
                references=["Gomes 2002", "Roosevelt 1999"]
            )
        ]
        
        self.sites = known_sites
        
        # Convert sites to knowledge documents
        for site in known_sites:
            doc_content = f"""
            Site Name: {site.name}
            Location: {site.coordinates}
            Culture: {site.culture}
            Time Period: {site.time_period}
            Site Type: {site.site_type}
            
            Characteristics:
            {chr(10).join(f"- {char}" for char in site.characteristics)}
            
            Discovery Method: {site.discovery_method}
            
            Significance: {site.significance}
            
            References: {', '.join(site.references)}
            """
            
            document = KnowledgeDocument(
                id=f"site_{site.name.lower().replace(' ', '_')}",
                title=f"Archaeological Site: {site.name}",
                content=doc_content.strip(),
                document_type="archaeological_site",
                metadata={
                    "coordinates": site.coordinates,
                    "culture": site.culture,
                    "time_period": site.time_period,
                    "site_type": site.site_type
                }
            )
            
            self.documents.append(document)
        
        # Add research methodology documents
        methodology_docs = [
            KnowledgeDocument(
                id="remote_sensing_methods",
                title="Remote Sensing Methods in Amazon Archaeology",
                content="""
                Remote sensing techniques for archaeological site detection in the Amazon:
                
                1. LiDAR (Light Detection and Ranging):
                - Penetrates forest canopy to reveal ground features
                - Excellent for detecting earthworks and mounds
                - High resolution topographic data
                
                2. Multispectral Satellite Imagery:
                - NDVI analysis reveals vegetation anomalies
                - Soil marks and crop marks indicate subsurface features
                - Seasonal variations help identify archaeological features
                
                3. Synthetic Aperture Radar (SAR):
                - Penetrates vegetation and works in cloudy conditions
                - Detects subtle topographic variations
                - Useful for monitoring deforestation and site exposure
                
                4. Hyperspectral Imaging:
                - Detailed spectral signatures of materials
                - Can detect specific minerals and soil types
                - Helps identify areas of human modification
                """,
                document_type="methodology",
                metadata={"topic": "remote_sensing", "application": "site_detection"}
            ),
            KnowledgeDocument(
                id="earthwork_detection",
                title="Earthwork Detection Techniques",
                content="""
                Techniques for detecting pre-Columbian earthworks in the Amazon:
                
                Geometric Patterns:
                - Circular, square, and octagonal earthworks
                - Concentric ring structures
                - Linear features and causeways
                
                Topographic Signatures:
                - Raised platforms and mounds
                - Defensive ditches and embankments
                - Water management features
                
                Vegetation Indicators:
                - Forest islands in savannas
                - Anthropogenic soils (Terra Preta)
                - Species composition changes
                
                Archaeological Indicators:
                - Ceramic concentrations
                - Lithic artifacts
                - Charcoal and dating materials
                """,
                document_type="methodology",
                metadata={"topic": "earthworks", "application": "detection"}
            )
        ]
        
        self.documents.extend(methodology_docs)

class RAGArchaeologist:
    """RAG system for archaeological analysis and site discovery"""
    
    def __init__(self, openai_api_key: str, vector_store_type: str = "memory"):
        self.embedding_generator = EmbeddingGenerator(openai_api_key)
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.knowledge_base = ArchaeologicalKnowledgeBase()
        
        # Initialize vector store
        if vector_store_type == "chroma" and CHROMADB_AVAILABLE:
            self.vector_store = ChromaVectorStore()
        else:
            self.vector_store = InMemoryVectorStore()
        
        self._build_vector_database()
    
    def _build_vector_database(self):
        """Build vector database from knowledge base"""
        print("Building vector database...")
        
        # Generate embeddings for all documents
        for doc in self.knowledge_base.documents:
            print(f"Processing: {doc.title}")
            doc.embedding = self.embedding_generator.generate_embedding(doc.content)
        
        # Add documents to vector store
        self.vector_store.add_documents(self.knowledge_base.documents)
        print(f"Vector database built with {len(self.knowledge_base.documents)} documents")
    
    def retrieve_relevant_context(self, query: str, k: int = 3) -> List[KnowledgeDocument]:
        """Retrieve relevant documents for a query"""
        # Generate embedding for query
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Search for similar documents
        similar_docs = self.vector_store.similarity_search(query_embedding, k)
        
        # Retrieve full documents
        relevant_docs = []
        for doc_id, similarity in similar_docs:
            doc = self.vector_store.get_document(doc_id)
            if doc:
                relevant_docs.append(doc)
        
        return relevant_docs
    
    def analyze_site_with_context(self, site_description: str, coordinates: Optional[tuple] = None) -> str:
        """Analyze a potential site using RAG-enhanced context"""
        
        # Retrieve relevant context
        relevant_docs = self.retrieve_relevant_context(site_description)
        
        # Build context string
        context_parts = []
        for doc in relevant_docs:
            context_parts.append(f"Title: {doc.title}\n{doc.content}")
        
        context_text = "\n\n---\n\n".join(context_parts)
        
        # Create enhanced prompt
        prompt = f"""
        You are an expert archaeologist specializing in Amazon basin pre-Columbian sites. 
        Use the following archaeological knowledge to analyze the potential site:

        CONTEXT:
        {context_text}

        SITE DESCRIPTION:
        {site_description}
        
        {"COORDINATES: " + str(coordinates) if coordinates else ""}

        Based on the context provided, please analyze:
        1. Likelihood this represents an archaeological site
        2. Cultural affiliation and time period
        3. Site type and function
        4. Comparison to known sites in the region
        5. Significance and research potential
        6. Recommended investigation methods

        Provide a detailed analysis with confidence levels for your assessments.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a leading expert in Amazon basin archaeology with comprehensive knowledge of pre-Columbian cultures."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error in analysis: {e}"
    
    def compare_with_known_sites(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Compare site features with known archaeological sites"""
        
        # Create comparison query
        feature_description = []
        for key, value in features.items():
            feature_description.append(f"{key}: {value}")
        
        query = f"Archaeological site with features: {'; '.join(feature_description)}"
        
        # Retrieve similar sites
        relevant_docs = self.retrieve_relevant_context(query, k=5)
        
        # Extract site information
        similar_sites = []
        for doc in relevant_docs:
            if doc.document_type == "archaeological_site":
                similar_sites.append({
                    "name": doc.metadata.get("site_name", doc.title),
                    "culture": doc.metadata.get("culture", "Unknown"),
                    "time_period": doc.metadata.get("time_period", "Unknown"),
                    "site_type": doc.metadata.get("site_type", "Unknown"),
                    "coordinates": doc.metadata.get("coordinates", None)
                })
        
        return {
            "query_features": features,
            "similar_sites": similar_sites,
            "total_comparisons": len(similar_sites)
        }
    
    def generate_investigation_plan(self, site_analysis: str, coordinates: tuple) -> str:
        """Generate detailed investigation plan for a potential site"""
        
        prompt = f"""
        Based on the following archaeological site analysis:
        
        {site_analysis}
        
        Site coordinates: {coordinates}
        
        Generate a comprehensive investigation plan including:
        
        1. REMOTE SENSING PHASE:
        - Recommended satellite imagery and resolution
        - Specific remote sensing techniques
        - Seasonal timing considerations
        
        2. GROUND SURVEY PHASE:
        - Survey methodology and grid layout
        - Equipment and personnel requirements
        - Site access and logistics
        
        3. EXCAVATION STRATEGY:
        - Priority areas for testing
        - Excavation methods and techniques
        - Sampling strategies
        
        4. ANALYSIS PLAN:
        - Laboratory analyses required
        - Dating methods and materials
        - Specialist consultations needed
        
        5. TIMELINE AND BUDGET:
        - Project phases and duration
        - Estimated costs and resources
        - Permit and legal requirements
        
        Provide specific, actionable recommendations based on the site characteristics.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an archaeological project director with extensive experience in Amazon basin fieldwork."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating investigation plan: {e}"

def main():
    """Demonstrate RAG system functionality"""
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    # Initialize RAG system
    print("Initializing Archaeological RAG System...")
    rag_archaeologist = RAGArchaeologist(api_key)
    
    # Example site analysis
    test_site = """
    Satellite imagery reveals circular earthwork structures approximately 200 meters in diameter,
    located on elevated terrain near a tributary of the Amazon River. The structures show
    geometric precision with multiple concentric rings and what appears to be radial divisions.
    Vegetation analysis indicates anthropogenic forest composition with Brazil nut trees and
    other useful species in higher concentrations than surrounding areas. LiDAR data suggests
    raised platforms within the circular structures.
    """
    
    test_coordinates = (-8.5, -63.2)  # Example coordinates in Rondônia
    
    print("\n" + "="*60)
    print("ARCHAEOLOGICAL SITE ANALYSIS")
    print("="*60)
    
    # Perform RAG-enhanced analysis
    analysis = rag_archaeologist.analyze_site_with_context(test_site, test_coordinates)
    print(analysis)
    
    print("\n" + "="*60)
    print("SITE COMPARISON")
    print("="*60)
    
    # Compare with known sites
    features = {
        "structure_type": "circular_earthworks",
        "diameter": "200_meters",
        "landscape_position": "elevated_terrain",
        "water_access": "river_tributary",
        "vegetation": "anthropogenic_forest"
    }
    
    comparison = rag_archaeologist.compare_with_known_sites(features)
    print(f"Found {comparison['total_comparisons']} similar sites:")
    for site in comparison['similar_sites']:
        print(f"- {site['name']} ({site['culture']}, {site['time_period']})")
    
    print("\n" + "="*60)
    print("INVESTIGATION PLAN")
    print("="*60)
    
    # Generate investigation plan
    investigation_plan = rag_archaeologist.generate_investigation_plan(analysis, test_coordinates)
    print(investigation_plan)
    
    # Save results
    results = {
        "site_description": test_site,
        "coordinates": test_coordinates,
        "analysis": analysis,
        "similar_sites": comparison,
        "investigation_plan": investigation_plan,
        "analysis_timestamp": datetime.now().isoformat()
    }
    
    output_file = '/home/myuser/OpenAI_to_Z_Challenge/rag_analysis_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()