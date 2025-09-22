import hashlib
import mimetypes
from pathlib import Path
from typing import Dict, Any, Optional
import magic  # python-magic for better MIME detection

def detect_mime_type(file_path: Path) -> str:
    """
    Detect MIME type of file
    
    Args:
        file_path: Path to file
        
    Returns:
        MIME type string
    """
    try:
        # Try python-magic first (more accurate)
        return magic.from_file(str(file_path), mime=True)
    except:
        # Fallback to mimetypes
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or "application/octet-stream"

def calculate_file_hashes(file_path: Path) -> Dict[str, str]:
    """
    Calculate multiple hash types for a file
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with hash types and values
    """
    hashes = {
        'md5': hashlib.md5(),
        'sha1': hashlib.sha1(),
        'sha256': hashlib.sha256(),
        'sha512': hashlib.sha512()
    }
    
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            for hash_obj in hashes.values():
                hash_obj.update(chunk)
    
    return {name: hash_obj.hexdigest() for name, hash_obj in hashes.items()}

def validate_evidence_id(evidence_id: str) -> bool:
    """
    Validate evidence ID format
    
    Args:
        evidence_id: Evidence identifier
        
    Returns:
        True if valid format
    """
    try:
        # Check if it's a valid UUID
        import uuid
        uuid.UUID(evidence_id)
        return True
    except ValueError:
        return False

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    import re
    # Remove or replace problematic characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Limit length
    if len(sanitized) > 255:
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        sanitized = name[:255-len(ext)-1] + '.' + ext if ext else name[:255]
    return sanitized