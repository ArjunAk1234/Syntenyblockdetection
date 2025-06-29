from flask import Flask, request, jsonify
import os
import sys
import json
import time
import tempfile
from typing import List, Tuple, Set, Optional
import numpy as np
from io import StringIO
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
# Configuration
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

class Node:
    """Node class for Suffix Tree implementation"""
    def __init__(self, start: int, end: int):
        self.children = {}
        self.suffix_link = None
        self.start = start
        self.end = end
        self.index = -1

class SuffixTree:
    """
    Suffix Tree implementation using Ukkonen's Algorithm
    Optimized for large genomic sequences
    """
    def __init__(self, text: str):
        self.text = text
        self.root = Node(-1, -1)
        self.size = len(text)
        self.active_node = self.root
        self.active_edge = -1
        self.active_length = 0
        self.remaining_suffix_count = 0
        self.leaf_end = -1
        self.last_new_node = None
        
        self.build_suffix_tree()
        
    def build_suffix_tree(self):
        """Build suffix tree using Ukkonen's algorithm"""
        for i in range(self.size):
            self.extend_suffix_tree(i)
    
    def edge_length(self, node: Node) -> int:
        """Calculate the length of an edge from current node"""
        return (node.end if node.end != -1 else self.leaf_end) - node.start + 1
    
    def walk_down(self, next_node: Node) -> bool:
        """Walk down the tree if active_length allows"""
        if self.active_length >= self.edge_length(next_node):
            self.active_edge += self.edge_length(next_node)
            self.active_length -= self.edge_length(next_node)
            self.active_node = next_node
            return True
        return False
    
    def extend_suffix_tree(self, pos: int):
        """Extend suffix tree with character at position pos (Ukkonen's algorithm core)"""
        self.leaf_end = pos
        self.remaining_suffix_count += 1
        self.last_new_node = None

        while self.remaining_suffix_count > 0:
            if self.active_length == 0:
                self.active_edge = pos

            if self.text[self.active_edge] not in self.active_node.children:
                # Create new leaf edge
                self.active_node.children[self.text[self.active_edge]] = Node(pos, -1)
                if self.last_new_node:
                    self.last_new_node.suffix_link = self.active_node
                    self.last_new_node = None
            else:
                next_node = self.active_node.children[self.text[self.active_edge]]
                if self.walk_down(next_node):
                    continue
                
                # Check if current character already exists on the edge
                if self.text[next_node.start + self.active_length] == self.text[pos]:
                    if self.last_new_node and self.active_node != self.root:
                        self.last_new_node.suffix_link = self.active_node
                        self.last_new_node = None
                    self.active_length += 1
                    break

                # Split the edge
                split_end = next_node.start + self.active_length - 1
                split = Node(next_node.start, split_end)
                self.active_node.children[self.text[self.active_edge]] = split
                split.children[self.text[pos]] = Node(pos, -1)
                next_node.start += self.active_length
                split.children[self.text[next_node.start]] = next_node
                
                if self.last_new_node:
                    self.last_new_node.suffix_link = split
                self.last_new_node = split

            self.remaining_suffix_count -= 1
            
            # Update active point
            if self.active_node == self.root and self.active_length > 0:
                self.active_length -= 1
                self.active_edge = pos - self.remaining_suffix_count + 1
            elif self.active_node != self.root:
                self.active_node = self.active_node.suffix_link if self.active_node.suffix_link else self.root

class SyntenyDetector:
    """
    Synteny Block Detection using Suffix Trees
    Handles large genome files efficiently - Memory only version
    """
    
    def __init__(self, min_block_length: int = 50):
        self.min_block_length = min_block_length
        self.chunk_size = 20000
        self.log_messages = []
        
    def log(self, message):
        """Add message to log for API response"""
        self.log_messages.append(message)
    
    def preprocess_fasta_content(self, file_content: str, filename: str = "genome") -> str:
        """
        Simple and efficient FASTA preprocessing from file content:
        - Remove headers (lines starting with >)
        - Remove all N sequences
        - Keep only valid DNA characters (ATCG)
        """
        self.log(f"Loading genome from {filename}...")
        
        sequence_parts = []
        lines = file_content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('>'):
                # Keep only ATCG characters, remove all N's and other characters
                clean_line = ''.join(c for c in line.upper() if c in 'ATCG')
                if clean_line:
                    sequence_parts.append(clean_line)
        
        dna_sequence = ''.join(sequence_parts)
        self.log(f"Loaded genome with {len(dna_sequence):,} bases (headers and N's filtered out)")
        return dna_sequence
    
    def collect_substrings(self, node: Node, text: str, path: str = '', 
                          results: Optional[List[str]] = None) -> List[str]:
        """
        Collect substrings from suffix tree that could be synteny blocks
        """
        if results is None:
            results = []
            
        if node.start != -1:
            end_pos = node.end + 1 if node.end != -1 else len(text)
            path += text[node.start:end_pos]
        
        # If this is an internal node with sufficient length, it's a potential synteny block
        if len(node.children) > 1 and len(path) >= self.min_block_length:
            # Check if it doesn't contain our separators
            if '#' not in path and '$' not in path:
                results.append(path)
        
        for child in node.children.values():
            self.collect_substrings(child, text, path, results)
        
        return results
    
    def merge_overlapping_blocks(self, blocks: List[str]) -> List[str]:
        """
        Merge overlapping synteny blocks and remove redundant ones
        """
        self.log("Merging overlapping blocks...")
        blocks = sorted(blocks, key=lambda x: (-len(x), x))
        merged = []
        
        for block in blocks:
            # Check if this block is a substring of any already merged block
            if not any(block in merged_block for merged_block in merged):
                merged.append(block)
        
        self.log(f"Merged {len(blocks)} blocks into {len(merged)} non-redundant blocks")
        return merged
    
    def detect_synteny_blocks_chunked(self, genome1_content: str, genome2_content: str, 
                                    genome1_name: str = "genome1", genome2_name: str = "genome2") -> List[str]:
        """
        Detect synteny blocks by processing genomes in chunks - memory only version
        """
        self.log("Starting synteny block detection...")
        
        # Preprocess genomes
        genome1 = self.preprocess_fasta_content(genome1_content, genome1_name)
        genome2 = self.preprocess_fasta_content(genome2_content, genome2_name)
        
        all_synteny_blocks = []
        chunk_num = 0
        
        # Process in chunks
        for i in range(0, min(len(genome1), len(genome2)), self.chunk_size):
            chunk_num += 1
            end_pos = min(i + self.chunk_size, min(len(genome1), len(genome2)))
            
            self.log(f"Processing chunk {chunk_num}: positions {i}-{end_pos}")
            
            # Extract chunks
            chunk1 = genome1[i:end_pos]
            chunk2 = genome2[i:end_pos]
            
            # Skip chunks that are too short
            if len(chunk1) < self.min_block_length or len(chunk2) < self.min_block_length:
                self.log(f"Skipping chunk {chunk_num} - too short")
                continue
            
            # Create combined sequence with separators
            combined_text = chunk1 + "#" + chunk2 + "$"
            
            self.log(f"Building suffix tree for chunk {chunk_num} (length: {len(combined_text)})")
            
            try:
                # Build suffix tree
                tree = SuffixTree(combined_text)
                
                # Find synteny blocks
                self.log(f"Searching for synteny blocks in chunk {chunk_num}...")
                chunk_blocks = self.collect_substrings(tree.root, combined_text)
                
                if chunk_blocks:
                    self.log(f"Found {len(chunk_blocks)} potential synteny blocks in chunk {chunk_num}")
                    all_synteny_blocks.extend(chunk_blocks)
                    
                    # Stop if we found significant synteny blocks
                    if len(chunk_blocks) > 0:
                        self.log(f"Found synteny blocks in chunk {chunk_num}. Analysis complete.")
                        break
                else:
                    self.log(f"No synteny blocks found in chunk {chunk_num}")
                    
            except Exception as e:
                self.log(f"Error processing chunk {chunk_num}: {e}")
                continue
        
        # Merge and filter results
        if all_synteny_blocks:
            final_blocks = self.merge_overlapping_blocks(all_synteny_blocks)
            return final_blocks
        else:
            self.log("No synteny blocks detected across all chunks")
            return []
    
    def analyze_synteny_blocks(self, blocks: List[str]) -> dict:
        """
        Analyze detected synteny blocks and provide statistics
        """
        if not blocks:
            return {"total_blocks": 0}
        
        analysis = {
            "total_blocks": len(blocks),
            "min_length": min(len(block) for block in blocks),
            "max_length": max(len(block) for block in blocks),
            "avg_length": sum(len(block) for block in blocks) / len(blocks),
            "total_conserved_sequence": sum(len(block) for block in blocks)
        }
        
        return analysis
    
    def generate_results_text(self, synteny_blocks: List[str], analysis: dict, 
                            genome1_name: str = "", genome2_name: str = "") -> str:
        """
        Generate comprehensive results text (instead of writing to file)
        """
        result_lines = []
        
        # Write header and metadata
        result_lines.append("=" * 80)
        result_lines.append("SYNTENY BLOCK DETECTION RESULTS")
        result_lines.append("=" * 80)
        result_lines.append(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        result_lines.append(f"Minimum Block Length Threshold: {self.min_block_length} bp")
        result_lines.append(f"Chunk Size Used: {self.chunk_size} bp")
        
        if genome1_name:
            result_lines.append(f"Genome 1: {genome1_name}")
        if genome2_name:
            result_lines.append(f"Genome 2: {genome2_name}")
        
        result_lines.append("")
        result_lines.append("=" * 80)
        result_lines.append("SUMMARY STATISTICS")
        result_lines.append("=" * 80)
        
        if synteny_blocks and analysis:
            result_lines.append(f"Total synteny blocks detected: {analysis['total_blocks']}")
            result_lines.append(f"Minimum block length: {analysis['min_length']} bp")
            result_lines.append(f"Maximum block length: {analysis['max_length']} bp")
            result_lines.append(f"Average block length: {analysis['avg_length']:.2f} bp")
            result_lines.append(f"Total conserved sequence: {analysis['total_conserved_sequence']} bp")
            
            # Calculate additional statistics
            lengths = [len(block) for block in synteny_blocks]
            lengths.sort(reverse=True)
            
            result_lines.append(f"Median block length: {lengths[len(lengths)//2]:.2f} bp")
            
            # Length distribution
            result_lines.append("")
            result_lines.append("LENGTH DISTRIBUTION:")
            result_lines.append("-" * 40)
            ranges = [(50, 100), (100, 500), (500, 1000), (1000, 5000), (5000, float('inf'))]
            for min_len, max_len in ranges:
                if max_len == float('inf'):
                    count = sum(1 for l in lengths if l >= min_len)
                    result_lines.append(f"{min_len}+ bp: {count} blocks")
                else:
                    count = sum(1 for l in lengths if min_len <= l < max_len)
                    result_lines.append(f"{min_len}-{max_len-1} bp: {count} blocks")
        else:
            result_lines.append("No synteny blocks detected.")
            result_lines.append("Possible reasons:")
            result_lines.append("- Low sequence similarity between genomes")
            result_lines.append("- Minimum block length threshold too high")
            result_lines.append("- Large evolutionary distance between species")
        
        # Write detailed block information
        if synteny_blocks:
            result_lines.append("")
            result_lines.append("=" * 80)
            result_lines.append("DETAILED SYNTENY BLOCKS")
            result_lines.append("=" * 80)
            
            for i, block in enumerate(synteny_blocks, 1):
                result_lines.append("")
                result_lines.append(f"Block #{i}:")
                result_lines.append(f"Length: {len(block)} bp")
                result_lines.append(f"GC Content: {(block.count('G') + block.count('C')) / len(block) * 100:.2f}%")
                
                # Write sequence in formatted blocks
                result_lines.append("Sequence:")
                for j in range(0, len(block), 80):  # 80 characters per line
                    line_num = j // 80 + 1
                    result_lines.append(f"{line_num:4d}: {block[j:j+80]}")
                
                result_lines.append("-" * 80)
                
            # Write FASTA format section
            result_lines.append("")
            result_lines.append("=" * 80)
            result_lines.append("SYNTENY BLOCKS IN FASTA FORMAT")
            result_lines.append("=" * 80)
            
            for i, block in enumerate(synteny_blocks, 1):
                result_lines.append(f">Synteny_Block_{i} length={len(block)}")
                # Write sequence in 80-character lines (standard FASTA format)
                for j in range(0, len(block), 80):
                    result_lines.append(f"{block[j:j+80]}")
        
        return "\n".join(result_lines)

# API Routes
@app.route('/')
def index():
    """API documentation endpoint"""
    return jsonify({
        "message": "Synteny Detection API - Memory Only Version",
        "version": "2.0",
        "endpoints": {
            "/analyze": {
                "method": "POST",
                "description": "Analyze synteny blocks between two genomes",
                "parameters": {
                    "genome1": "FASTA file (multipart/form-data)",
                    "genome2": "FASTA file (multipart/form-data)", 
                    "min_length": "Minimum block length (optional, default: 100)"
                },
                "returns": "JSON with analysis results including full text report"
            }
        },
        "note": "Files are processed in memory only - no files are saved on server"
    })

@app.route('/analyze', methods=['POST'])
def analyze_synteny():
    """Main API endpoint for synteny analysis - memory only version"""
    try:
        # Check if files are present in request
        if 'genome1' not in request.files or 'genome2' not in request.files:
            return jsonify({
                "error": "Both genome1 and genome2 files are required",
                "status": "error"
            }), 400
        
        genome1_file = request.files['genome1']
        genome2_file = request.files['genome2']
        
        # Check if files are selected
        if genome1_file.filename == '' or genome2_file.filename == '':
            return jsonify({
                "error": "Please select both genome files",
                "status": "error"
            }), 400
        
        # Get minimum length parameter
        min_length = int(request.form.get('min_length', 100))
        
        # Read file contents directly into memory
        try:
            genome1_content = genome1_file.read().decode('utf-8')
            genome2_content = genome2_file.read().decode('utf-8')
        except UnicodeDecodeError:
            return jsonify({
                "error": "Files must be text-based FASTA files (UTF-8 encoding)",
                "status": "error"
            }), 400
        
        # Initialize detector
        detector = SyntenyDetector(min_block_length=min_length)
        
        # Start analysis
        start_time = time.time()
        
        # Detect synteny blocks
        synteny_blocks = detector.detect_synteny_blocks_chunked(
            genome1_content, 
            genome2_content,
            genome1_file.filename,
            genome2_file.filename
        )
        
        # Analyze results
        analysis = detector.analyze_synteny_blocks(synteny_blocks) if synteny_blocks else {"total_blocks": 0}
        
        # Generate text report
        text_report = detector.generate_results_text(
            synteny_blocks, 
            analysis,
            genome1_file.filename,
            genome2_file.filename
        )
        
        end_time = time.time()
        analysis_time = end_time - start_time
        
        # Prepare response
        response_data = {
            "status": "success",
            "analysis_time_seconds": round(analysis_time, 2),
            "parameters": {
                "genome1_filename": genome1_file.filename,
                "genome2_filename": genome2_file.filename,
                "min_block_length": min_length
            },
            "results": analysis,
            "log_messages": detector.log_messages,
            "detailed_report": text_report
        }
        
        # Add synteny blocks info if found
        if synteny_blocks:
            response_data["synteny_blocks"] = []
            for i, block in enumerate(synteny_blocks[:10], 1):  # Return first 10 blocks
                response_data["synteny_blocks"].append({
                    "block_id": i,
                    "length": len(block),
                    "gc_content": round((block.count('G') + block.count('C')) / len(block) * 100, 2),
                    "sequence_preview": block[:100] + ("..." if len(block) > 100 else ""),
                    "full_sequence": block  # Include full sequence in JSON
                })
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "message": "An error occurred during analysis"
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "service": "Synteny Detection API - Memory Only Version"
    })

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({
        "error": "File too large. Maximum size is 500MB per file.",
        "status": "error"
    }), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        "error": "Internal server error",
        "status": "error"
    }), 500

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')  # Allow all
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response
