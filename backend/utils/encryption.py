"""
Encryption utilities for API keys and sensitive data.
Uses Fernet symmetric encryption for secure storage.
"""

import base64
import os
from pathlib import Path
from cryptography.fernet import Fernet
from typing import Optional


# Cache the encryption key to ensure consistency across the application lifecycle
_encryption_key_cache: Optional[bytes] = None


def get_encryption_key() -> bytes:
    """
    Get or generate the encryption key for API keys.
    
    Priority order:
    1. ENCRYPTION_KEY environment variable (for production/secrets management)
    2. .encryption_key file in the app directory (auto-generated on first run)
    3. Generate new key and save to file (first-time setup)
    
    The key is cached to ensure consistency during the application lifecycle.
    """
    global _encryption_key_cache
    
    if _encryption_key_cache is not None:
        return _encryption_key_cache
    
    # 1. Check environment variable first
    key_str = os.getenv('ENCRYPTION_KEY')
    if key_str:
        try:
            _encryption_key_cache = key_str.encode()
            print("✅ Using ENCRYPTION_KEY from environment variable")
            return _encryption_key_cache
        except Exception:
            raise ValueError("Invalid ENCRYPTION_KEY format")
    

    # remove this in production for security reasons
    # 2. Check for persistent key file
    key_file = Path("/app/.encryption_key_data/key")
    if key_file.exists():
        try:
            with open(key_file, 'rb') as f:
                _encryption_key_cache = f.read().strip()
            print(f"✅ Loaded encryption key from {key_file}")
            return _encryption_key_cache
        except Exception as e:
            print(f"⚠️  Failed to load key from {key_file}: {e}")
    
    # 3. Generate new key and persist it
    key = Fernet.generate_key()
    try:
        key_file.parent.mkdir(parents=True, exist_ok=True)
        with open(key_file, 'wb') as f:
            f.write(key)
        # Set restrictive permissions (owner read/write only)
        key_file.chmod(0o600)
        print(f"🔑 Generated new encryption key and saved to {key_file}")
        print(f"⚠️  IMPORTANT: Back up this key! If lost, encrypted API keys cannot be recovered.")
    except Exception as e:
        print(f"⚠️  Could not save key to file: {e}. Key will only persist for this session.")
    
    _encryption_key_cache = key
    return key


def encrypt_api_key(api_key: str) -> str:
    """
    Encrypt an API key for storage.
    Returns base64-encoded encrypted string.
    """
    if not api_key:
        return ""
    
    key = get_encryption_key()
    fernet = Fernet(key)
    encrypted = fernet.encrypt(api_key.encode())
    return base64.b64encode(encrypted).decode()


def decrypt_api_key(encrypted_key: str) -> str:
    """
    Decrypt an API key from storage.
    Takes base64-encoded encrypted string, returns plain text.
    """
    if not encrypted_key:
        return ""
    
    try:
        key = get_encryption_key()
        fernet = Fernet(key)
        encrypted_bytes = base64.b64decode(encrypted_key.encode())
        decrypted = fernet.decrypt(encrypted_bytes)
        return decrypted.decode()
    except Exception as e:
        raise ValueError(f"Failed to decrypt API key: {e}")


def mask_api_key(api_key: str) -> str:
    """
    Mask an API key for display purposes.
    Shows first few and last few characters.
    """
    if not api_key or len(api_key) < 8:
        return "***"
    
    if api_key.startswith('sk-'):
        # OpenAI key format
        return f"sk-...{api_key[-6:]}"
    elif api_key.startswith('sk-ant-'):
        # Anthropic key format
        return f"sk-ant-...{api_key[-6:]}"
    else:
        # Generic masking
        return f"{api_key[:4]}...{api_key[-4:]}"

