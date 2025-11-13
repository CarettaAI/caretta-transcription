"""
JWT authentication utilities for WebSocket connections.
Verifies Supabase JWT tokens without caching.
"""
from __future__ import annotations

import jwt
import requests
from typing import Dict, Any, Optional
from fastapi import WebSocket, WebSocketException, status

from .config import logger, SUPABASE_ISSUER, SUPABASE_JWKS, EXPECT_AUD


class JWKSClient:
    """Simple JWKS client that fetches keys from a URL."""
    
    def __init__(self, jwks_url: str):
        self.jwks_url = jwks_url
    
    def get_signing_keys(self) -> Dict[str, Any]:
        """Fetch JWKS and return as dict keyed by 'kid'."""
        try:
            response = requests.get(self.jwks_url, timeout=5)
            response.raise_for_status()
            jwks = response.json()
            
            keys = {}
            for key in jwks.get("keys", []):
                kid = key.get("kid")
                if kid:
                    keys[kid] = key
            return keys
        except Exception as e:
            logger.error("Failed to fetch JWKS: %s", e)
            raise


def verify_jwt_token(token: str) -> Dict[str, Any]:
    """
    Verify a JWT token against Supabase JWKS.
    
    Args:
        token: The JWT token string
        
    Returns:
        The decoded JWT payload
        
    Raises:
        jwt.InvalidTokenError: If token is invalid
        Exception: If JWKS fetch fails or verification fails
    """
    if not SUPABASE_ISSUER or not SUPABASE_JWKS:
        raise ValueError("SUPABASE_ISSUER and SUPABASE_JWKS must be configured")
    
    # Fetch JWKS
    jwks_client = JWKSClient(SUPABASE_JWKS)
    signing_keys = jwks_client.get_signing_keys()
    
    # Decode header to get kid
    unverified_header = jwt.get_unverified_header(token)
    kid = unverified_header.get("kid")
    
    if not kid:
        raise jwt.InvalidTokenError("Token missing 'kid' in header")
    
    if kid not in signing_keys:
        raise jwt.InvalidTokenError(f"Unknown key ID: {kid}")
    
    # Get the signing key
    jwk = signing_keys[kid]
    
    # Convert JWK to PEM for PyJWT
    public_key = jwt.get_algorithm_by_name("RS256").from_jwk(jwk)
    
    # Verify and decode the token
    payload = jwt.decode(
        token,
        public_key,
        algorithms=["RS256"],
        issuer=SUPABASE_ISSUER,
        audience=EXPECT_AUD,
        options={
            "verify_signature": True,
            "verify_exp": True,
            "verify_iat": True,
            "verify_aud": True,
            "verify_iss": True,
        }
    )
    
    return payload


async def verify_websocket_auth(websocket: WebSocket) -> Dict[str, Any]:
    """
    Verify JWT authentication for WebSocket connection.
    
    Extracts token from:
    1. Query parameter: ?token=...
    2. Authorization header: Bearer <token>
    
    Args:
        websocket: The FastAPI WebSocket connection
        
    Returns:
        The decoded JWT payload
        
    Raises:
        WebSocketException: If authentication fails
    """
    # Try query parameter first
    token = websocket.query_params.get("token")
    
    # Try Authorization header if no query param
    if not token:
        auth_header = websocket.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
    
    if not token:
        logger.warning("WebSocket connection attempt without token")
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Authentication required"
        )
    
    try:
        payload = verify_jwt_token(token)
        logger.info("WebSocket authenticated: user_id=%s", payload.get("sub"))
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("WebSocket connection with expired token")
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Token expired"
        )
    except jwt.InvalidTokenError as e:
        logger.warning("WebSocket connection with invalid token: %s", e)
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Invalid token"
        )
    except Exception as e:
        logger.error("WebSocket authentication error: %s", e)
        raise WebSocketException(
            code=status.WS_1011_INTERNAL_ERROR,
            reason="Authentication failed"
        )
