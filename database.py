import sqlite3
import threading
from cryptography.fernet import Fernet
import os
from dotenv import load_dotenv
import logging

load_dotenv()

class APIKeyManager:
    _instance = None
    _lock = threading.Lock()
    _local = threading.local()

    def __init__(self):
        self.encryption_key = os.getenv('ENCRYPTION_KEY').encode()
        self.cipher_suite = Fernet(self.encryption_key)
        self.db_path = 'keys.db'  # 本地数据库路径
        self._init_db()

    def _get_conn(self):
        """Get thread-local database connection"""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(self.db_path)
        return self._local.conn

    def _init_db(self):
        """Initialize database with required tables"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS api_keys
                (user_id TEXT, service TEXT, api_key TEXT,
                PRIMARY KEY (user_id, service))
            ''')
            conn.commit()
            conn.close()

    def add_api_key(self, user_id: str, service: str, api_key: str):
        """Add or update an API key for a user"""
        encrypted_key = self.cipher_suite.encrypt(api_key.encode())
        conn = self._get_conn()
        c = conn.cursor()
        c.execute('''
            INSERT OR REPLACE INTO api_keys (user_id, service, api_key)
            VALUES (?, ?, ?)
        ''', (user_id, service, encrypted_key))
        conn.commit()

    def get_api_key(self, user_id: str, service: str) -> str:
        """Get a user's API key for a service"""
        conn = self._get_conn()
        c = conn.cursor()
        c.execute('''
            SELECT api_key FROM api_keys
            WHERE user_id = ? AND service = ?
        ''', (user_id, service))
        result = c.fetchone()
        if result:
            decrypted_key = self.cipher_suite.decrypt(result[0])
            return decrypted_key.decode()
        return None

    def delete_api_key(self, user_id: str, service: str):
        """Delete a user's API key for a service"""
        conn = self._get_conn()
        c = conn.cursor()
        c.execute('''
            DELETE FROM api_keys
            WHERE user_id = ? AND service = ?
        ''', (user_id, service))
        conn.commit()

    def has_active_key(self, user_id: str, service: str) -> bool:
        """Check if user has an active key"""
        return self.get_api_key(user_id, service) is not None

    def revoke_key(self, user_id: str, service: str):
        """Revoke (delete) a user's API key"""
        self.delete_api_key(user_id, service)

    def get_all_keys(self, service: str) -> dict:
        """Get all active keys for a service"""
        conn = self._get_conn()
        c = conn.cursor()
        c.execute('''
            SELECT user_id, api_key FROM api_keys
            WHERE service = ?
        ''', (service,))
        results = c.fetchall()
        keys = {}
        for user_id, encrypted_key in results:
            decrypted_key = self.cipher_suite.decrypt(encrypted_key)
            keys[user_id] = decrypted_key.decode()
        return keys

    def __del__(self):
        if hasattr(self, '_local') and hasattr(self._local, 'conn'):
            self._local.conn.close() 