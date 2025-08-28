#!/usr/bin/env python3
"""
Memory Encryption Manager for MeRNSTA Sovereign Mode
Provides AES-GCM encryption at rest for databases and memory storage.
"""

import os
import sys
import json
import sqlite3
import logging
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from contextlib import contextmanager
from datetime import datetime
import threading

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system.sovereign_crypto import get_sovereign_crypto
from storage.audit_logger import get_audit_logger

logger = logging.getLogger(__name__)


class EncryptedDatabase:
    """
    Encrypted database wrapper that transparently encrypts/decrypts database files.
    
    Provides the same interface as a regular SQLite database but automatically
    handles encryption at rest using AES-GCM.
    """
    
    def __init__(self, db_path: str, auto_encrypt: bool = True):
        """Initialize encrypted database wrapper."""
        self.db_path = Path(db_path)
        self.auto_encrypt = auto_encrypt
        self.crypto = get_sovereign_crypto()
        self.audit_logger = get_audit_logger("memory_encryption")
        
        # Track if database is currently encrypted
        self._is_encrypted = self._check_if_encrypted()
        self._connection_count = 0
        self._lock = threading.RLock()
        
        # Temporary decrypted path
        self._temp_path: Optional[Path] = None
        
        logger.debug(f"Initialized encrypted database: {db_path} (encrypted: {self._is_encrypted})")
    
    def _check_if_encrypted(self) -> bool:
        """Check if database file is currently encrypted."""
        if not self.db_path.exists():
            return False
        
        try:
            # Try to read as JSON (encrypted format)
            with open(self.db_path, 'r') as f:
                data = json.load(f)
                return isinstance(data, dict) and 'ciphertext' in data
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Not JSON, likely unencrypted SQLite
            return False
        except Exception:
            # File exists but can't be read - assume encrypted
            return True
    
    @contextmanager
    def connection(self, **kwargs):
        """Get a database connection with automatic encryption/decryption."""
        with self._lock:
            self._connection_count += 1
            
            try:
                # Decrypt database if needed
                if self._is_encrypted:
                    db_path = self._decrypt_database()
                else:
                    db_path = self.db_path
                
                # Create SQLite connection
                conn = sqlite3.connect(str(db_path), **kwargs)
                
                try:
                    yield conn
                finally:
                    conn.close()
                    
            finally:
                self._connection_count -= 1
                
                # Re-encrypt if this was the last connection and auto-encrypt is enabled
                if self._connection_count == 0 and self.auto_encrypt:
                    self._encrypt_database()
    
    def _decrypt_database(self) -> Path:
        """Decrypt database to temporary location."""
        if not self._is_encrypted:
            return self.db_path
        
        try:
            # Read encrypted data
            with open(self.db_path, 'r') as f:
                encrypted_data = json.load(f)
            
            # Decrypt
            db_data = self.crypto.decrypt_data(encrypted_data)
            
            # Write to temporary file
            if not self._temp_path:
                self._temp_path = Path(tempfile.mktemp(suffix='.db'))
            
            with open(self._temp_path, 'wb') as f:
                f.write(db_data)
            
            # Log decryption
            asyncio.create_task(self.audit_logger.log_event({
                "event_type": "memory_decrypted",
                "agent_id": "memory_encryption",
                "database_path": str(self.db_path),
                "temp_path": str(self._temp_path)
            }))
            
            logger.debug(f"Decrypted database {self.db_path} to {self._temp_path}")
            return self._temp_path
            
        except Exception as e:
            logger.error(f"Failed to decrypt database {self.db_path}: {e}")
            raise
    
    def _encrypt_database(self):
        """Encrypt database from temporary location back to main location."""
        if not self._temp_path or not self._temp_path.exists():
            return
        
        try:
            # Read decrypted data
            with open(self._temp_path, 'rb') as f:
                db_data = f.read()
            
            # Encrypt
            encrypted_data = self.crypto.encrypt_data(db_data)
            
            # Write encrypted data
            with open(self.db_path, 'w') as f:
                json.dump(encrypted_data, f, indent=2)
            
            # Clean up temporary file
            self._temp_path.unlink()
            self._temp_path = None
            
            # Update state
            self._is_encrypted = True
            
            # Log encryption
            asyncio.create_task(self.audit_logger.log_event({
                "event_type": "memory_encrypted",
                "agent_id": "memory_encryption",
                "database_path": str(self.db_path)
            }))
            
            logger.debug(f"Encrypted database {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to encrypt database {self.db_path}: {e}")
            raise
    
    def encrypt_now(self) -> bool:
        """Manually encrypt the database immediately."""
        with self._lock:
            if self._is_encrypted:
                logger.info(f"Database {self.db_path} is already encrypted")
                return True
            
            if self._connection_count > 0:
                logger.warning(f"Cannot encrypt {self.db_path} - active connections exist")
                return False
            
            try:
                # Read unencrypted database
                with open(self.db_path, 'rb') as f:
                    db_data = f.read()
                
                # Create backup
                backup_path = self.db_path.with_suffix('.backup')
                shutil.copy2(self.db_path, backup_path)
                
                # Encrypt
                encrypted_data = self.crypto.encrypt_data(db_data)
                
                # Write encrypted data
                with open(self.db_path, 'w') as f:
                    json.dump(encrypted_data, f, indent=2)
                
                self._is_encrypted = True
                
                # Log encryption
                asyncio.create_task(self.audit_logger.log_event({
                    "event_type": "database_encrypted",
                    "agent_id": "memory_encryption",
                    "database_path": str(self.db_path),
                    "backup_path": str(backup_path)
                }))
                
                logger.info(f"Successfully encrypted database: {self.db_path}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to encrypt database {self.db_path}: {e}")
                return False
    
    def decrypt_now(self) -> bool:
        """Manually decrypt the database immediately."""
        with self._lock:
            if not self._is_encrypted:
                logger.info(f"Database {self.db_path} is already decrypted")
                return True
            
            if self._connection_count > 0:
                logger.warning(f"Cannot decrypt {self.db_path} - active connections exist")
                return False
            
            try:
                # Read encrypted data
                with open(self.db_path, 'r') as f:
                    encrypted_data = json.load(f)
                
                # Create backup
                backup_path = self.db_path.with_suffix('.encrypted_backup')
                shutil.copy2(self.db_path, backup_path)
                
                # Decrypt
                db_data = self.crypto.decrypt_data(encrypted_data)
                
                # Write decrypted data
                with open(self.db_path, 'wb') as f:
                    f.write(db_data)
                
                self._is_encrypted = False
                
                # Log decryption
                asyncio.create_task(self.audit_logger.log_event({
                    "event_type": "database_decrypted",
                    "agent_id": "memory_encryption",
                    "database_path": str(self.db_path),
                    "backup_path": str(backup_path)
                }))
                
                logger.info(f"Successfully decrypted database: {self.db_path}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to decrypt database {self.db_path}: {e}")
                return False
    
    def is_encrypted(self) -> bool:
        """Check if database is currently encrypted."""
        return self._is_encrypted
    
    def verify_integrity(self) -> Dict[str, Any]:
        """Verify database integrity."""
        try:
            if self._is_encrypted:
                # Test decryption
                with self.connection() as conn:
                    conn.execute("SELECT 1").fetchone()
                status = "encrypted_accessible"
            else:
                # Test direct access
                with sqlite3.connect(str(self.db_path)) as conn:
                    conn.execute("SELECT 1").fetchone()
                status = "unencrypted_accessible"
            
            return {
                "status": status,
                "encrypted": self._is_encrypted,
                "accessible": True,
                "path": str(self.db_path),
                "size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0
            }
            
        except Exception as e:
            return {
                "status": "error",
                "encrypted": self._is_encrypted,
                "accessible": False,
                "error": str(e),
                "path": str(self.db_path)
            }
    
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, '_temp_path') and self._temp_path and self._temp_path.exists():
            try:
                self._temp_path.unlink()
            except Exception:
                pass


class MemoryEncryptionManager:
    """
    Centralized manager for memory encryption across all MeRNSTA databases.
    
    Handles:
    - Bulk encryption/decryption of configured databases
    - Database registry and lifecycle management
    - Encryption key rotation
    - System-wide encryption status monitoring
    """
    
    def __init__(self):
        """Initialize memory encryption manager."""
        self.crypto = get_sovereign_crypto()
        self.config = self.crypto.sovereign_config.get("memory_encryption", {})
        self.audit_logger = get_audit_logger("memory_encryption")
        
        # Database registry
        self.encrypted_dbs: Dict[str, EncryptedDatabase] = {}
        self._initialized = False
        
        logger.info("Memory encryption manager initialized")
    
    def initialize(self):
        """Initialize encryption for all configured databases."""
        if self._initialized:
            return
        
        if not self.config.get("enabled", False):
            logger.info("Memory encryption is disabled in configuration")
            return
        
        database_list = self.config.get("encrypted_databases", [])
        
        for db_name in database_list:
            try:
                self.register_database(db_name)
            except Exception as e:
                logger.error(f"Failed to register database {db_name}: {e}")
        
        self._initialized = True
        logger.info(f"Memory encryption initialized for {len(self.encrypted_dbs)} databases")
    
    def register_database(self, db_path: str) -> EncryptedDatabase:
        """Register a database for encryption management."""
        abs_path = os.path.abspath(db_path)
        
        if abs_path in self.encrypted_dbs:
            return self.encrypted_dbs[abs_path]
        
        # Create encrypted database wrapper
        encrypted_db = EncryptedDatabase(
            abs_path, 
            auto_encrypt=self.config.get("enabled", False)
        )
        
        self.encrypted_dbs[abs_path] = encrypted_db
        
        # Log registration
        asyncio.create_task(self.audit_logger.log_event({
            "event_type": "database_registered",
            "agent_id": "memory_encryption",
            "database_path": abs_path,
            "auto_encrypt": encrypted_db.auto_encrypt
        }))
        
        logger.info(f"Registered database for encryption: {db_path}")
        return encrypted_db
    
    def get_database(self, db_path: str) -> Optional[EncryptedDatabase]:
        """Get encrypted database wrapper."""
        abs_path = os.path.abspath(db_path)
        return self.encrypted_dbs.get(abs_path)
    
    def encrypt_all_databases(self) -> Dict[str, bool]:
        """Encrypt all registered databases."""
        results = {}
        
        for db_path, encrypted_db in self.encrypted_dbs.items():
            try:
                results[db_path] = encrypted_db.encrypt_now()
            except Exception as e:
                logger.error(f"Failed to encrypt {db_path}: {e}")
                results[db_path] = False
        
        # Log bulk encryption
        asyncio.create_task(self.audit_logger.log_event({
            "event_type": "bulk_encryption",
            "agent_id": "memory_encryption",
            "results": results,
            "total_databases": len(results),
            "successful": sum(results.values())
        }))
        
        return results
    
    def decrypt_all_databases(self) -> Dict[str, bool]:
        """Decrypt all registered databases."""
        results = {}
        
        for db_path, encrypted_db in self.encrypted_dbs.items():
            try:
                results[db_path] = encrypted_db.decrypt_now()
            except Exception as e:
                logger.error(f"Failed to decrypt {db_path}: {e}")
                results[db_path] = False
        
        # Log bulk decryption
        asyncio.create_task(self.audit_logger.log_event({
            "event_type": "bulk_decryption",
            "agent_id": "memory_encryption",
            "results": results,
            "total_databases": len(results),
            "successful": sum(results.values())
        }))
        
        return results
    
    def get_encryption_status(self) -> Dict[str, Any]:
        """Get system-wide encryption status."""
        if not self._initialized:
            self.initialize()
        
        database_status = {}
        encrypted_count = 0
        accessible_count = 0
        
        for db_path, encrypted_db in self.encrypted_dbs.items():
            integrity = encrypted_db.verify_integrity()
            database_status[db_path] = integrity
            
            if integrity.get("encrypted", False):
                encrypted_count += 1
            if integrity.get("accessible", False):
                accessible_count += 1
        
        return {
            "encryption_enabled": self.config.get("enabled", False),
            "total_databases": len(self.encrypted_dbs),
            "encrypted_databases": encrypted_count,
            "accessible_databases": accessible_count,
            "encryption_algorithm": self.config.get("algorithm", "aes-gcm"),
            "key_size": self.config.get("key_size", 256),
            "database_status": database_status,
            "system_status": "healthy" if accessible_count == len(self.encrypted_dbs) else "degraded"
        }
    
    def rotate_encryption_keys(self) -> bool:
        """Rotate encryption keys for all databases."""
        try:
            # Generate new master key
            old_crypto = self.crypto
            self.crypto = get_sovereign_crypto()  # This will generate new keys
            
            # Re-encrypt all databases with new keys
            results = {}
            for db_path, encrypted_db in self.encrypted_dbs.items():
                try:
                    # Decrypt with old key
                    if encrypted_db.is_encrypted():
                        encrypted_db.crypto = old_crypto
                        encrypted_db.decrypt_now()
                    
                    # Re-encrypt with new key
                    encrypted_db.crypto = self.crypto
                    results[db_path] = encrypted_db.encrypt_now()
                    
                except Exception as e:
                    logger.error(f"Failed to rotate keys for {db_path}: {e}")
                    results[db_path] = False
            
            # Log key rotation
            asyncio.create_task(self.audit_logger.log_event({
                "event_type": "encryption_key_rotation",
                "agent_id": "memory_encryption",
                "results": results,
                "successful": sum(results.values()),
                "total_databases": len(results)
            }))
            
            success = all(results.values())
            if success:
                logger.info("Successfully rotated encryption keys for all databases")
            else:
                logger.warning("Key rotation partially failed - some databases may be inaccessible")
            
            return success
            
        except Exception as e:
            logger.error(f"Encryption key rotation failed: {e}")
            return False
    
    def emergency_decrypt_all(self) -> Dict[str, bool]:
        """Emergency decryption of all databases (for recovery)."""
        logger.warning("EMERGENCY DECRYPTION INITIATED")
        
        # Log emergency action
        asyncio.create_task(self.audit_logger.log_event({
            "event_type": "emergency_decryption",
            "agent_id": "memory_encryption",
            "timestamp": datetime.now().isoformat(),
            "reason": "emergency_recovery"
        }))
        
        return self.decrypt_all_databases()
    
    def seal_system(self) -> Dict[str, Any]:
        """Seal the system by encrypting all databases and securing keys."""
        if not self.config.get("enabled", False):
            return {"status": "encryption_disabled", "sealed": False}
        
        try:
            # Encrypt all databases
            encryption_results = self.encrypt_all_databases()
            
            # Generate new identity for additional security
            self.crypto.generate_identity(force_new=True)
            
            # Log sealing
            asyncio.create_task(self.audit_logger.log_event({
                "event_type": "system_sealed",
                "agent_id": "memory_encryption",
                "encryption_results": encryption_results,
                "timestamp": datetime.now().isoformat()
            }))
            
            sealed = all(encryption_results.values())
            
            return {
                "status": "sealed" if sealed else "partially_sealed",
                "sealed": sealed,
                "encryption_results": encryption_results,
                "identity_regenerated": True
            }
            
        except Exception as e:
            logger.error(f"System sealing failed: {e}")
            return {"status": "seal_failed", "sealed": False, "error": str(e)}


# Global memory encryption manager
_memory_encryption_manager = None

def get_memory_encryption_manager() -> MemoryEncryptionManager:
    """Get or create global memory encryption manager."""
    global _memory_encryption_manager
    if _memory_encryption_manager is None:
        _memory_encryption_manager = MemoryEncryptionManager()
        _memory_encryption_manager.initialize()
    return _memory_encryption_manager


# Convenience functions for database access
def get_encrypted_connection(db_path: str, **kwargs):
    """Get an encrypted database connection."""
    manager = get_memory_encryption_manager()
    encrypted_db = manager.get_database(db_path)
    
    if encrypted_db is None:
        # Register database on first access
        encrypted_db = manager.register_database(db_path)
    
    return encrypted_db.connection(**kwargs)


def ensure_database_encrypted(db_path: str) -> bool:
    """Ensure a specific database is encrypted."""
    manager = get_memory_encryption_manager()
    encrypted_db = manager.get_database(db_path)
    
    if encrypted_db is None:
        encrypted_db = manager.register_database(db_path)
    
    return encrypted_db.encrypt_now()


def ensure_database_decrypted(db_path: str) -> bool:
    """Ensure a specific database is decrypted."""
    manager = get_memory_encryption_manager()
    encrypted_db = manager.get_database(db_path)
    
    if encrypted_db is None:
        encrypted_db = manager.register_database(db_path)
    
    return encrypted_db.decrypt_now()