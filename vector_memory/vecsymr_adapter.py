# vector_memory/vecsymr_adapter.py

"""
VecSymR Adapter for MeRNSTA

This adapter integrates VecSymR's human-like analogical mapping capabilities
for cognitive vector processing and symbolic reasoning.
"""

import subprocess
import tempfile
import logging
import os
from typing import List

def vecsymr_vectorize(text: str) -> List[float]:
    """
    Vectorize text using VecSymR analogical mapping.
    
    Args:
        text: Input text to vectorize
        
    Returns:
        List of floats representing the analogical vector
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt') as f:
            f.write(text)
            f.flush()
            
            # Call R script for VecSymR encoding
            result = subprocess.check_output([
                "Rscript", "external/vecsymr/R/encode_vector.R", f.name
            ], timeout=30)
            
            # Parse the result
            vector_str = result.decode().strip()
            if vector_str:
                vector = [float(x) for x in vector_str.split()]
                logging.info(f"[VecSymR] Generated vector of size {len(vector)} for text: '{text[:50]}...'")
                return vector
            else:
                raise ValueError("Empty result from R script")
                
    except subprocess.TimeoutExpired:
        logging.error(f"[VecSymR] Timeout processing text: '{text[:50]}...'")
        return [0.0] * 512
    except subprocess.CalledProcessError as e:
        logging.error(f"[VecSymR] R script failed: {e}")
        return [0.0] * 512
    except Exception as e:
        logging.error(f"[VecSymR] Failed: {e}")
        return [0.0] * 512
    finally:
        # Clean up temporary file
        try:
            if 'f' in locals():
                os.unlink(f.name)
        except:
            pass

def vecsymr_vectorize_batch(texts: List[str]) -> List[List[float]]:
    """
    Vectorize multiple texts efficiently.
    
    Args:
        texts: List of input texts
        
    Returns:
        List of vectors (each a list of floats)
    """
    return [vecsymr_vectorize(text) for text in texts]

def check_vecsymr_dependencies() -> dict:
    """
    Check if VecSymR dependencies are available.
    
    Returns:
        Dictionary with status information
    """
    status = {
        "rscript_available": False,
        "vecsymr_installed": False,
        "encode_script_exists": False,
        "status": "not_ready"
    }
    
    try:
        # Check if Rscript is available
        subprocess.check_output(["Rscript", "--version"], timeout=5)
        status["rscript_available"] = True
    except:
        pass
    
    try:
        # Check if encode script exists
        script_path = "external/vecsymr/R/encode_vector.R"
        if os.path.exists(script_path):
            status["encode_script_exists"] = True
    except:
        pass
    
    # TODO: Add check for vecsymr R package installation
    # This would require running: R -e "packageVersion('vecsymr')"
    
    if status["rscript_available"] and status["encode_script_exists"]:
        status["status"] = "ready"
    elif status["rscript_available"]:
        status["status"] = "missing_script"
    else:
        status["status"] = "missing_rscript"
    
    return status

def get_vecsymr_info() -> dict:
    """Get information about VecSymR configuration."""
    deps = check_vecsymr_dependencies()
    return {
        "backend": "VecSymR",
        "description": "Human-like analogical mapping for cognitive reasoning",
        "vector_size": 512,  # This would be determined by actual implementation
        "dependencies": deps,
        "note": "Requires R and vecsymr package installation"
    }