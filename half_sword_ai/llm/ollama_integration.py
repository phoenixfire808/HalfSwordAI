"""
Ollama Qwen Integration Module
Handles communication with Ollama API for high-level decision making
Enhanced with caching, retry logic, performance tracking, and comprehensive error debugging
"""
import requests
import json
import time
import hashlib
import traceback
import sys
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from half_sword_ai.config import config
import logging

# Use centralized logging setup from agent.py
logger = logging.getLogger(__name__)

# Error tracking for debugging
class LLMErrorTracker:
    """Tracks and categorizes LLM errors for debugging"""
    def __init__(self):
        self.error_history: deque = deque(maxlen=50)
        self.error_counts: Dict[str, int] = {}
        self.last_error: Optional[Dict[str, Any]] = None
    
    def record_error(self, error_type: str, error_msg: str, context: Dict[str, Any]):
        """Record an error with full context"""
        error_entry = {
            'timestamp': time.time(),
            'type': error_type,
            'message': error_msg,
            'context': context.copy(),
            'traceback': traceback.format_exc()
        }
        self.error_history.append(error_entry)
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        self.last_error = error_entry
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors"""
        return {
            'total_errors': len(self.error_history),
            'error_counts': self.error_counts.copy(),
            'last_error': self.last_error,
            'recent_errors': list(self.error_history)[-10:]
        }

class OllamaQwenAgent:
    """
    Enhanced integration with Ollama Qwen model for strategic decision making
    Features:
    - Retry logic with exponential backoff
    - Response caching for similar queries
    - Performance metrics tracking
    - Context-aware conversation history
    - Robust JSON parsing with multiple fallbacks
    - Rate limiting protection
    """
    
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or config.OLLAMA_BASE_URL
        self.model = model or config.OLLAMA_MODEL
        self.session = requests.Session()
        self.session.timeout = config.OLLAMA_TIMEOUT
        
        # Conversation history with context management
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history_length = 10  # Keep last 10 exchanges
        
        # Response caching
        self.response_cache: Dict[str, Tuple[Dict[str, Any], float]] = {}
        self.cache_ttl = 2.0  # Cache responses for 2 seconds
        self.cache_max_size = 50
        
        # Performance metrics
        self.metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'cache_hits': 0,
            'average_latency': 0.0,
            'latencies': deque(maxlen=100),
            'last_query_time': 0.0
        }
        
        # Retry configuration
        self.max_retries = 2
        self.retry_delays = [0.1, 0.3]  # Exponential backoff delays
        
        # Rate limiting
        self.min_query_interval = 0.5  # Minimum time between queries
        self.last_query_time = 0.0
        
        # Fallback models
        self.fallback_models = [
            config.OLLAMA_MODEL,
            "qwen2.5-7b-instruct",
            "qwen2.5-3b-instruct",
            "qwen2.5-1.5b-instruct"
        ]
        self.current_model_index = 0
        
        # Error tracking and debugging
        self.error_tracker = LLMErrorTracker()
        self.debug_mode = config.DETAILED_LOGGING
        self.request_id_counter = 0
        
    def _make_request(self, endpoint: str, data: Dict[str, Any], retry_count: int = 0, request_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make HTTP request to Ollama API with retry logic, timeout protection, and detailed error debugging
        
        Args:
            endpoint: API endpoint to call
            data: Request payload
            retry_count: Current retry attempt number
            request_context: Additional context for error tracking
        """
        self.request_id_counter += 1
        request_id = self.request_id_counter
        url = f"{self.base_url}/{endpoint}"
        start_time = time.time()
        model_name = data.get('model', 'unknown')
        
        # Build detailed context for error tracking
        error_context = {
            'request_id': request_id,
            'endpoint': endpoint,
            'url': url,
            'model': model_name,
            'retry_count': retry_count,
            'base_url': self.base_url,
            'timeout': config.OLLAMA_TIMEOUT,
            'prompt_length': len(data.get('prompt', '')),
            'options': data.get('options', {}),
            **(request_context or {})
        }
        
        if self.debug_mode:
            logger.debug(f"[LLM Request #{request_id}] Starting request to {endpoint}")
            logger.debug(f"[LLM Request #{request_id}] Model: {model_name}, Retry: {retry_count}/{self.max_retries}")
            logger.debug(f"[LLM Request #{request_id}] URL: {url}")
            logger.debug(f"[LLM Request #{request_id}] Payload size: {len(json.dumps(data))} bytes")
        
        try:
            response = self.session.post(url, json=data, timeout=config.OLLAMA_TIMEOUT)
            latency = time.time() - start_time
            
            # Log response details
            if self.debug_mode:
                logger.debug(f"[LLM Request #{request_id}] Response received in {latency:.3f}s")
                logger.debug(f"[LLM Request #{request_id}] Status code: {response.status_code}")
                logger.debug(f"[LLM Request #{request_id}] Response headers: {dict(response.headers)}")
            
            # Check for HTTP errors
            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as http_err:
                error_msg = f"HTTP {response.status_code}: {http_err}"
                self.error_tracker.record_error(
                    'HTTP_ERROR',
                    error_msg,
                    {
                        **error_context,
                        'status_code': response.status_code,
                        'response_text': response.text[:500],
                        'response_headers': dict(response.headers)
                    }
                )
                logger.error(f"[LLM Request #{request_id}] ❌ HTTP Error: {error_msg}")
                logger.error(f"[LLM Request #{request_id}] Response body: {response.text[:500]}")
                self.metrics['failed_queries'] += 1
                
                # Retry on 5xx errors
                if response.status_code >= 500 and retry_count < self.max_retries:
                    delay = self.retry_delays[min(retry_count, len(self.retry_delays) - 1)]
                    logger.warning(f"[LLM Request #{request_id}] Retrying after {delay}s (server error)")
                    time.sleep(delay)
                    return self._make_request(endpoint, data, retry_count + 1, request_context)
                
                return {"error": error_msg, "status_code": response.status_code, "strategy": "defensive", "confidence": 0.5}
            
            # Parse JSON response
            try:
                result = response.json()
            except json.JSONDecodeError as json_err:
                error_msg = f"JSON decode error: {json_err}"
                self.error_tracker.record_error(
                    'JSON_DECODE_ERROR',
                    error_msg,
                    {
                        **error_context,
                        'response_text': response.text[:1000],
                        'response_length': len(response.text)
                    }
                )
                logger.error(f"[LLM Request #{request_id}] ❌ JSON Parse Error: {error_msg}")
                logger.error(f"[LLM Request #{request_id}] Response text (first 500 chars): {response.text[:500]}")
                self.metrics['failed_queries'] += 1
                return {"error": error_msg, "strategy": "defensive", "confidence": 0.5}
            
            # Check for error in response body
            if isinstance(result, dict) and "error" in result:
                error_msg = result.get("error", "Unknown error in response")
                self.error_tracker.record_error(
                    'API_ERROR',
                    error_msg,
                    {
                        **error_context,
                        'api_error': error_msg,
                        'full_response': result
                    }
                )
                logger.error(f"[LLM Request #{request_id}] ❌ API Error in response: {error_msg}")
                logger.error(f"[LLM Request #{request_id}] Full response: {json.dumps(result, indent=2)}")
                self.metrics['failed_queries'] += 1
                return result
            
            # Track successful request
            self.metrics['successful_queries'] += 1
            self.metrics['latencies'].append(latency)
            self._update_average_latency()
            
            if self.debug_mode:
                response_length = len(json.dumps(result))
                logger.debug(f"[LLM Request #{request_id}] ✅ Success - Response size: {response_length} bytes")
                if 'response' in result:
                    logger.debug(f"[LLM Request #{request_id}] Response preview: {result.get('response', '')[:200]}...")
            
            return result
            
        except requests.exceptions.Timeout:
            latency = time.time() - start_time
            self.metrics['failed_queries'] += 1
            
            error_msg = f"Request timeout after {latency:.3f}s (limit: {config.OLLAMA_TIMEOUT}s)"
            self.error_tracker.record_error(
                'TIMEOUT',
                error_msg,
                {
                    **error_context,
                    'actual_latency': latency,
                    'timeout_limit': config.OLLAMA_TIMEOUT
                }
            )
            logger.error(f"[LLM Request #{request_id}] ⏱️ Timeout: {error_msg}")
            
            # Retry with exponential backoff
            if retry_count < self.max_retries:
                delay = self.retry_delays[min(retry_count, len(self.retry_delays) - 1)]
                logger.warning(f"[LLM Request #{request_id}] Retrying in {delay}s (attempt {retry_count + 1}/{self.max_retries})")
                time.sleep(delay)
                return self._make_request(endpoint, data, retry_count + 1, request_context)
            
            logger.error(f"[LLM Request #{request_id}] Max retries reached - returning default strategy")
            return {"error": "timeout", "strategy": "defensive", "confidence": 0.5}
            
        except requests.exceptions.ConnectionError as conn_err:
            latency = time.time() - start_time
            self.metrics['failed_queries'] += 1
            
            error_msg = f"Connection error: {str(conn_err)}"
            self.error_tracker.record_error(
                'CONNECTION_ERROR',
                error_msg,
                {
                    **error_context,
                    'connection_error': str(conn_err),
                    'error_type': type(conn_err).__name__
                }
            )
            logger.error(f"[LLM Request #{request_id}] 🔌 Connection Error: {error_msg}")
            logger.error(f"[LLM Request #{request_id}] Check if Ollama is running at {self.base_url}")
            logger.error(f"[LLM Request #{request_id}] Full error: {traceback.format_exc()}")
            
            # Retry on connection errors
            if retry_count < self.max_retries:
                delay = self.retry_delays[min(retry_count, len(self.retry_delays) - 1)]
                logger.warning(f"[LLM Request #{request_id}] Retrying in {delay}s (attempt {retry_count + 1}/{self.max_retries})")
                time.sleep(delay)
                return self._make_request(endpoint, data, retry_count + 1, request_context)
            
            return {"error": error_msg, "strategy": "defensive", "confidence": 0.5}
            
        except requests.exceptions.RequestException as e:
            latency = time.time() - start_time
            self.metrics['failed_queries'] += 1
            
            error_msg = f"Request exception: {str(e)}"
            error_type = type(e).__name__
            self.error_tracker.record_error(
                'REQUEST_EXCEPTION',
                error_msg,
                {
                    **error_context,
                    'exception_type': error_type,
                    'exception_message': str(e),
                    'full_traceback': traceback.format_exc()
                }
            )
            logger.error(f"[LLM Request #{request_id}] ❌ Request Exception ({error_type}): {error_msg}")
            logger.error(f"[LLM Request #{request_id}] Full traceback:\n{traceback.format_exc()}")
            
            # Retry on specific recoverable errors
            if retry_count < self.max_retries and isinstance(e, (requests.exceptions.ChunkedEncodingError, requests.exceptions.InvalidURL)):
                delay = self.retry_delays[min(retry_count, len(self.retry_delays) - 1)]
                logger.warning(f"[LLM Request #{request_id}] Retrying in {delay}s: {e}")
                time.sleep(delay)
                return self._make_request(endpoint, data, retry_count + 1, request_context)
            
            return {"error": error_msg, "strategy": "defensive", "confidence": 0.5}
            
        except Exception as e:
            latency = time.time() - start_time
            self.metrics['failed_queries'] += 1
            
            error_msg = f"Unexpected error: {str(e)}"
            error_type = type(e).__name__
            self.error_tracker.record_error(
                'UNEXPECTED_ERROR',
                error_msg,
                {
                    **error_context,
                    'exception_type': error_type,
                    'exception_message': str(e),
                    'full_traceback': traceback.format_exc()
                }
            )
            logger.error(f"[LLM Request #{request_id}] 💥 Unexpected Error ({error_type}): {error_msg}")
            logger.error(f"[LLM Request #{request_id}] Full traceback:\n{traceback.format_exc()}")
            return {"error": error_msg, "strategy": "defensive", "confidence": 0.5}
    
    def _update_average_latency(self):
        """Update average latency metric"""
        if self.metrics['latencies']:
            self.metrics['average_latency'] = sum(self.metrics['latencies']) / len(self.metrics['latencies'])
    
    def _get_cache_key(self, game_state: Dict[str, Any], frame_description: str) -> str:
        """Generate cache key from game state and frame description"""
        # Create a simplified state representation for caching
        cache_data = {
            'health': round(game_state.get('health', 0) / 10) * 10,  # Round to nearest 10
            'stamina': round(game_state.get('stamina', 0) / 10) * 10,
            'enemy_health': round(game_state.get('enemy_health', 0) / 10) * 10,
            'is_dead': game_state.get('is_dead', False),
            'enemy_dead': game_state.get('enemy_dead', False),
            'combat_state': game_state.get('combat_state', 'unknown'),
            'frame_desc': frame_description[:100] if frame_description else ''
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available and not expired"""
        if cache_key in self.response_cache:
            cached_response, cached_time = self.response_cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                self.metrics['cache_hits'] += 1
                return cached_response.copy()
            else:
                # Remove expired cache entry
                del self.response_cache[cache_key]
        return None
    
    def _cache_response(self, cache_key: str, response: Dict[str, Any]):
        """Cache response with size limit"""
        # Remove oldest entries if cache is full
        if len(self.response_cache) >= self.cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = min(self.response_cache.keys(), 
                           key=lambda k: self.response_cache[k][1])
            del self.response_cache[oldest_key]
        
        self.response_cache[cache_key] = (response.copy(), time.time())
    
    def analyze_game_state(self, game_state: Dict[str, Any], frame_description: str = "") -> Dict[str, Any]:
        """
        Use Qwen to analyze current game state and provide strategic guidance
        Enhanced with caching, rate limiting, and improved error handling
        
        Args:
            game_state: Dictionary with health, stamina, enemy info, etc.
            frame_description: Optional text description of visual frame
            
        Returns:
            Dictionary with strategic recommendations
        """
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_query_time < self.min_query_interval:
            # Return cached or default response if rate limited
            cache_key = self._get_cache_key(game_state, frame_description)
            cached = self._get_cached_response(cache_key)
            if cached:
                return cached
            # Return last successful response or default
            if self.conversation_history:
                last_response = self.conversation_history[-1].get('response', {})
                if isinstance(last_response, dict) and 'strategy' in last_response:
                    return last_response
        
        # Check cache first
        cache_key = self._get_cache_key(game_state, frame_description)
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response
        
        self.metrics['total_queries'] += 1
        self.last_query_time = current_time
        self.metrics['last_query_time'] = current_time
        
        # Build enhanced prompt with context
        prompt = self._build_state_analysis_prompt(game_state, frame_description)
        
        # Log the prompt being sent (only in debug mode to reduce spam)
        if config.DETAILED_LOGGING:
            logger.debug("="*60)
            logger.debug("📤 SENDING TO OLLAMA QWEN:")
            logger.debug("="*60)
            logger.debug(f"Prompt (first 500 chars):\n{prompt[:500]}...")
            logger.debug(f"Full game state: {json.dumps(game_state, indent=2)}")
        
        # Try with current model, fallback to others if needed
        response = None
        request_context = {
            'game_state_summary': {
                'health': game_state.get('health', 0),
                'stamina': game_state.get('stamina', 0),
                'enemy_health': game_state.get('enemy_health', 0),
                'combat_state': game_state.get('combat_state', 'unknown')
            },
            'frame_description_length': len(frame_description),
            'cache_key': cache_key
        }
        
        models_tried = []
        for attempt in range(len(self.fallback_models)):
            model_to_use = self.fallback_models[(self.current_model_index + attempt) % len(self.fallback_models)]
            models_tried.append(model_to_use)
            
            if self.debug_mode:
                logger.debug(f"[LLM] Attempting model {attempt + 1}/{len(self.fallback_models)}: {model_to_use}")
            
            response = self._make_request("api/generate", {
                "model": model_to_use,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temp for more consistent tactical decisions
                    "top_p": 0.9,
                    "num_predict": 200  # Limit response length for faster processing
                }
            }, request_context=request_context)
            
            # If successful, update current model index
            if "error" not in response or attempt == len(self.fallback_models) - 1:
                if "error" not in response:
                    if model_to_use != self.fallback_models[self.current_model_index]:
                        logger.info(f"[LLM] Switched to model: {model_to_use} (was: {self.fallback_models[self.current_model_index]})")
                    self.current_model_index = (self.current_model_index + attempt) % len(self.fallback_models)
                break
        
        if "error" in response:
            error_msg = response.get('error', 'Unknown error')
            error_type = response.get('error_type', 'API_ERROR')
            
            self.error_tracker.record_error(
                error_type,
                error_msg,
                {
                    **request_context,
                    'models_tried': models_tried,
                    'final_model': models_tried[-1] if models_tried else 'none',
                    'response': response
                }
            )
            
            logger.error(f"[LLM] ❌ All models failed after {len(models_tried)} attempts")
            logger.error(f"[LLM] Error: {error_msg}")
            logger.error(f"[LLM] Models tried: {', '.join(models_tried)}")
            
            # Provide diagnostic information
            if not self.check_health():
                logger.error(f"[LLM] 🔍 DIAGNOSTIC: Ollama service appears to be down")
                logger.error(f"[LLM] 🔍 Check: Is Ollama running? Try: ollama serve")
                logger.error(f"[LLM] 🔍 Check: Is the base URL correct? Current: {self.base_url}")
            
            error_response = {"strategy": "defensive", "confidence": 0.5, "reasoning": f"API error: {error_msg}"}
            self._cache_response(cache_key, error_response)
            return error_response
        
        reasoning = response.get("response", "")
        
        # Validate response structure
        if not reasoning or not isinstance(reasoning, str):
            error_msg = f"Invalid response format: {type(reasoning)}"
            self.error_tracker.record_error(
                'INVALID_RESPONSE',
                error_msg,
                {
                    **request_context,
                    'response_type': type(reasoning).__name__,
                    'response_value': str(reasoning)[:500],
                    'full_response': response
                }
            )
            logger.error(f"[LLM] ❌ Invalid response format: expected string, got {type(reasoning)}")
            logger.error(f"[LLM] Response structure: {json.dumps(response, indent=2)[:500]}")
            return self._get_default_strategy(game_state)
        
        # Log the response received (only in debug mode)
        if config.DETAILED_LOGGING:
            logger.debug("="*60)
            logger.debug("📥 RECEIVED FROM OLLAMA QWEN:")
            logger.debug("="*60)
            logger.debug(f"Raw response length: {len(reasoning)} chars")
            logger.debug(f"Raw response:\n{reasoning}")
        
        try:
            parsed = self._parse_strategic_response(reasoning, game_state)
        except Exception as parse_err:
            error_msg = f"Failed to parse response: {str(parse_err)}"
            self.error_tracker.record_error(
                'PARSE_ERROR',
                error_msg,
                {
                    **request_context,
                    'raw_response': reasoning[:1000],
                    'response_length': len(reasoning),
                    'parse_exception': str(parse_err),
                    'parse_traceback': traceback.format_exc()
                }
            )
            logger.error(f"[LLM] ❌ Parse Error: {error_msg}")
            logger.error(f"[LLM] Raw response that failed to parse:\n{reasoning[:500]}")
            logger.error(f"[LLM] Parse traceback:\n{traceback.format_exc()}")
            parsed = self._get_default_strategy(game_state)
        
        # Update conversation history
        self._update_conversation_history(game_state, frame_description, parsed)
        
        # Cache the response
        self._cache_response(cache_key, parsed)
        
        if config.DETAILED_LOGGING:
            logger.debug(f"Parsed strategy: {parsed.get('strategy')} (confidence: {parsed.get('confidence'):.2f})")
            logger.debug(f"Reasoning: {parsed.get('reasoning', '')[:200]}")
            logger.debug("="*60)
        
        return parsed
    
    def _update_conversation_history(self, game_state: Dict[str, Any], 
                                    frame_description: str, response: Dict[str, Any]):
        """Update conversation history with context"""
        # Add to history
        self.conversation_history.append({
            'state': game_state.copy(),
            'frame_desc': frame_description,
            'response': response.copy(),
            'timestamp': time.time()
        })
        
        # Trim history if too long
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def _build_state_analysis_prompt(self, game_state: Dict[str, Any], frame_description: str) -> str:
        """Build enhanced prompt for Qwen with context from conversation history"""
        # Build context from recent history
        context_summary = ""
        if len(self.conversation_history) > 0:
            recent = self.conversation_history[-3:]  # Last 3 exchanges
            context_summary = "\nRecent Context:\n"
            for i, hist in enumerate(recent):
                prev_strategy = hist.get('response', {}).get('strategy', 'unknown')
                context_summary += f"- Previous strategy {i+1}: {prev_strategy}\n"
        
        # Enhanced game state information
        yolo_info = ""
        if 'yolo_detections' in game_state:
            yolo_det = game_state['yolo_detections']
            yolo_info = f"""
YOLO Vision Analysis:
- Enemies Detected: {yolo_det.get('enemy_count', 0)}
- Threat Level: {yolo_det.get('threat_level', 'unknown')}
- Nearest Enemy Distance: {yolo_det.get('nearest_enemy_distance', 'N/A')}px
"""
        
        prompt = f"""You are an expert Half Sword combat AI analyzing a real-time combat situation.

Current Game State:
- Player Health: {game_state.get('health', 0):.1f}/100
- Player Stamina: {game_state.get('stamina', 0):.1f}/100
- Enemy Health: {game_state.get('enemy_health', 0):.1f}/100
- Is Dead: {game_state.get('is_dead', False)}
- Enemy Dead: {game_state.get('enemy_dead', False)}
- Player Position: {game_state.get('position', 'unknown')}
- Combat State: {game_state.get('combat_state', 'unknown')}
{yolo_info}
Visual Context: {frame_description if frame_description else 'Frame captured, analyzing...'}
{context_summary}
Based on this state, provide a strategic recommendation in JSON format:
{{
    "strategy": "aggressive|defensive|evasive|counter",
    "primary_action": "attack|block|dodge|reposition",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "tactical_notes": "specific movement or timing advice"
}}

CRITICAL: Respond ONLY with valid JSON, no additional text or markdown formatting."""
        return prompt
    
    def _parse_strategic_response(self, response_text: str, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse Qwen's response into structured format with multiple fallback strategies
        Enhanced with detailed error tracking for each parsing strategy
        """
        if not response_text or not response_text.strip():
            if self.debug_mode:
                logger.warning("[LLM Parse] Empty response text, using default strategy")
            return self._get_default_strategy(game_state)
        
        parse_errors = []
        
        # Strategy 1: Try direct JSON parsing
        try:
            parsed = json.loads(response_text.strip())
            if isinstance(parsed, dict):
                if self.debug_mode:
                    logger.debug("[LLM Parse] ✅ Strategy 1 (direct JSON) succeeded")
                return self._validate_and_normalize_strategy(parsed, game_state)
            else:
                parse_errors.append(f"Strategy 1: Expected dict, got {type(parsed).__name__}")
        except json.JSONDecodeError as e:
            parse_errors.append(f"Strategy 1 (direct JSON): {str(e)}")
            if self.debug_mode:
                logger.debug(f"[LLM Parse] Strategy 1 failed: {e}")
        
        # Strategy 2: Extract JSON from markdown code blocks
        try:
            import re
            json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
            matches = re.findall(json_pattern, response_text, re.DOTALL)
            if matches:
                parsed = json.loads(matches[0])
                if isinstance(parsed, dict):
                    if self.debug_mode:
                        logger.debug("[LLM Parse] ✅ Strategy 2 (markdown JSON) succeeded")
                    return self._validate_and_normalize_strategy(parsed, game_state)
                else:
                    parse_errors.append(f"Strategy 2: Expected dict, got {type(parsed).__name__}")
            else:
                parse_errors.append("Strategy 2: No markdown code blocks found")
        except (json.JSONDecodeError, AttributeError, re.error) as e:
            parse_errors.append(f"Strategy 2 (markdown JSON): {str(e)}")
            if self.debug_mode:
                logger.debug(f"[LLM Parse] Strategy 2 failed: {e}")
        
        # Strategy 3: Extract JSON object from text
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                parsed = json.loads(json_str)
                if isinstance(parsed, dict):
                    if self.debug_mode:
                        logger.debug("[LLM Parse] ✅ Strategy 3 (extracted JSON) succeeded")
                    return self._validate_and_normalize_strategy(parsed, game_state)
                else:
                    parse_errors.append(f"Strategy 3: Expected dict, got {type(parsed).__name__}")
            else:
                parse_errors.append(f"Strategy 3: No JSON braces found (start: {json_start}, end: {json_end})")
        except (json.JSONDecodeError, ValueError) as e:
            parse_errors.append(f"Strategy 3 (extracted JSON): {str(e)}")
            if self.debug_mode:
                logger.debug(f"[LLM Parse] Strategy 3 failed: {e}")
        
        # Strategy 4: Intelligent keyword-based parsing
        if self.debug_mode:
            logger.warning(f"[LLM Parse] All JSON parsing strategies failed, falling back to keyword parsing")
            logger.warning(f"[LLM Parse] Parse errors: {'; '.join(parse_errors)}")
            logger.warning(f"[LLM Parse] Response text: {response_text[:300]}...")
        
        return self._parse_from_keywords(response_text, game_state)
    
    def _validate_and_normalize_strategy(self, parsed: Dict[str, Any], game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize parsed strategy response"""
        # Validate strategy value
        valid_strategies = ["aggressive", "defensive", "evasive", "counter"]
        strategy = parsed.get("strategy", "defensive").lower()
        if strategy not in valid_strategies:
            # Try to infer from other fields
            if "aggressive" in str(parsed.get("primary_action", "")).lower():
                strategy = "aggressive"
            elif "dodge" in str(parsed.get("primary_action", "")).lower():
                strategy = "evasive"
            else:
                strategy = "defensive"
        
        # Validate primary action
        valid_actions = ["attack", "block", "dodge", "reposition"]
        primary_action = parsed.get("primary_action", "block").lower()
        if primary_action not in valid_actions:
            primary_action = "block"
        
        # Validate confidence
        try:
            confidence = float(parsed.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
        except (ValueError, TypeError):
            confidence = 0.5
        
        # Adjust strategy based on game state if needed
        if game_state.get('is_dead', False):
            strategy = "defensive"
            confidence = 0.3
        
        return {
            "strategy": strategy,
            "primary_action": primary_action,
            "confidence": confidence,
            "reasoning": str(parsed.get("reasoning", ""))[:300],
            "tactical_notes": str(parsed.get("tactical_notes", ""))[:200]
        }
    
    def _parse_from_keywords(self, response_text: str, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Parse strategy from keywords when JSON parsing fails"""
        text_lower = response_text.lower()
        
        # Determine strategy
        strategy = "defensive"
        confidence = 0.6
        
        if any(word in text_lower for word in ["aggressive", "attack", "offensive", "strike"]):
            strategy = "aggressive"
            confidence = 0.7
        elif any(word in text_lower for word in ["evasive", "dodge", "retreat", "escape"]):
            strategy = "evasive"
            confidence = 0.65
        elif any(word in text_lower for word in ["counter", "parry", "riposte"]):
            strategy = "counter"
            confidence = 0.7
        
        # Determine primary action
        primary_action = "block"
        if "attack" in text_lower or "strike" in text_lower:
            primary_action = "attack"
        elif "dodge" in text_lower or "evade" in text_lower:
            primary_action = "dodge"
        elif "reposition" in text_lower or "move" in text_lower:
            primary_action = "reposition"
        
        return {
            "strategy": strategy,
            "primary_action": primary_action,
            "confidence": confidence,
            "reasoning": response_text[:300],
            "tactical_notes": ""
        }
    
    def _get_default_strategy(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get default strategy based on game state"""
        # Intelligent defaults based on state
        if game_state.get('is_dead', False):
            return {
                "strategy": "defensive",
                "primary_action": "block",
                "confidence": 0.3,
                "reasoning": "Player is dead, defaulting to defensive",
                "tactical_notes": ""
            }
        
        health = game_state.get('health', 100)
        enemy_health = game_state.get('enemy_health', 100)
        
        if health < 30:
            strategy = "evasive"
            primary_action = "dodge"
            confidence = 0.8
        elif enemy_health < 30 and health > 50:
            strategy = "aggressive"
            primary_action = "attack"
            confidence = 0.7
        else:
            strategy = "defensive"
            primary_action = "block"
            confidence = 0.6
        
        return {
            "strategy": strategy,
            "primary_action": primary_action,
            "confidence": confidence,
            "reasoning": "Default strategy based on game state",
            "tactical_notes": ""
        }
    
    def interpret_visual_frame(self, frame_summary: str) -> Dict[str, Any]:
        """
        Use Qwen's vision capabilities to interpret game frame
        Enhanced with caching and better error handling
        Note: Requires Qwen-VL model for full vision support
        """
        # Check cache
        cache_key = hashlib.md5(f"visual_{frame_summary[:100]}".encode()).hexdigest()
        cached = self._get_cached_response(cache_key)
        if cached:
            return cached
        
        prompt = f"""Analyze this Half Sword combat frame description:
{frame_summary}

Identify:
1. Enemy position and weapon orientation
2. Threat level (high/medium/low)
3. Recommended defensive or offensive action
4. Timing cues (is enemy winding up an attack?)

Respond in JSON format:
{{
    "threat_level": "high|medium|low",
    "action": "attack|block|dodge|observe",
    "enemy_orientation": "description",
    "timing_cue": "description"
}}

CRITICAL: Respond ONLY with valid JSON."""
        
        response = self._make_request("api/generate", {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.4,
                "num_predict": 150
            }
        })
        
        if "error" in response:
            default_response = {"threat_level": "medium", "action": "block", "enemy_orientation": "unknown"}
            self._cache_response(cache_key, default_response)
            return default_response
        
        parsed = self._parse_visual_analysis(response.get("response", ""))
        self._cache_response(cache_key, parsed)
        return parsed
    
    def _parse_visual_analysis(self, response_text: str) -> Dict[str, Any]:
        """Parse visual analysis response with JSON fallback"""
        if not response_text:
            return {"threat_level": "medium", "action": "block", "enemy_orientation": "unknown"}
        
        # Try JSON parsing first
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                parsed = json.loads(response_text[json_start:json_end])
                if isinstance(parsed, dict):
                    return {
                        "threat_level": parsed.get("threat_level", "medium").lower(),
                        "action": parsed.get("action", "block").lower(),
                        "enemy_orientation": parsed.get("enemy_orientation", "unknown"),
                        "timing_cue": parsed.get("timing_cue", "")
                    }
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback to keyword parsing
        text_lower = response_text.lower()
        threat_level = "medium"
        if any(word in text_lower for word in ["high threat", "attacking", "dangerous", "immediate"]):
            threat_level = "high"
        elif any(word in text_lower for word in ["low threat", "idle", "safe", "distant"]):
            threat_level = "low"
        
        action = "block"
        if threat_level == "high":
            action = "block"
        elif "dodge" in text_lower or "evade" in text_lower:
            action = "dodge"
        elif "attack" in text_lower:
            action = "attack"
        
        return {
            "threat_level": threat_level,
            "action": action,
            "enemy_orientation": "unknown",
            "timing_cue": ""
        }
    
    def generate_training_insight(self, episode_data: Dict[str, Any]) -> str:
        """
        Use Qwen to analyze training episode and provide insights
        Enhanced with better prompt engineering
        """
        # Build context from metrics
        metrics_summary = ""
        if self.metrics['total_queries'] > 0:
            success_rate = (self.metrics['successful_queries'] / self.metrics['total_queries']) * 100
            metrics_summary = f"""
LLM Performance:
- Success Rate: {success_rate:.1f}%
- Average Latency: {self.metrics['average_latency']:.3f}s
- Cache Hit Rate: {(self.metrics['cache_hits'] / max(1, self.metrics['total_queries'])) * 100:.1f}%
"""
        
        prompt = f"""Analyze this Half Sword training episode and provide actionable insights:

Episode Metrics:
- Duration: {episode_data.get('duration', 0):.1f}s
- Outcome: {episode_data.get('outcome', 'unknown')}
- Damage Taken: {episode_data.get('damage_taken', 0)}
- Damage Dealt: {episode_data.get('damage_dealt', 0)}
- Human Interventions: {episode_data.get('interventions', 0)}
{metrics_summary}

Provide 2-3 specific, actionable recommendations for improving the agent's combat performance.
Focus on tactical adjustments, timing improvements, or strategic changes."""
        
        response = self._make_request("api/generate", {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.5,  # Slightly higher for more creative insights
                "num_predict": 300
            }
        })
        
        if "error" in response:
            return "Continue training and collect more data. LLM analysis unavailable."
        
        return response.get("response", "Continue training and collect more data.")
    
    def check_health(self) -> bool:
        """Check if Ollama service is available with timeout and detailed diagnostics"""
        try:
            health_url = f"{self.base_url}/api/tags"
            if self.debug_mode:
                logger.debug(f"[LLM Health] Checking Ollama at {health_url}")
            
            response = self.session.get(health_url, timeout=3)
            
            if response.status_code == 200:
                if self.debug_mode:
                    logger.debug(f"[LLM Health] ✅ Ollama is healthy")
                return True
            else:
                self.error_tracker.record_error(
                    'HEALTH_CHECK_FAILED',
                    f"Health check returned status {response.status_code}",
                    {
                        'url': health_url,
                        'status_code': response.status_code,
                        'response_text': response.text[:500]
                    }
                )
                logger.error(f"[LLM Health] ❌ Health check failed: HTTP {response.status_code}")
                logger.error(f"[LLM Health] Response: {response.text[:200]}")
                return False
                
        except requests.exceptions.Timeout:
            self.error_tracker.record_error(
                'HEALTH_CHECK_TIMEOUT',
                "Health check timed out",
                {'url': f"{self.base_url}/api/tags", 'timeout': 3}
            )
            logger.error(f"[LLM Health] ⏱️ Health check timed out after 3s")
            logger.error(f"[LLM Health] Ollama may be overloaded or not responding")
            return False
            
        except requests.exceptions.ConnectionError as e:
            self.error_tracker.record_error(
                'HEALTH_CHECK_CONNECTION_ERROR',
                f"Connection error: {str(e)}",
                {
                    'url': f"{self.base_url}/api/tags",
                    'error': str(e),
                    'suggestion': 'Check if Ollama is running: ollama serve'
                }
            )
            logger.error(f"[LLM Health] 🔌 Connection error: {e}")
            logger.error(f"[LLM Health] 💡 Suggestion: Start Ollama with 'ollama serve'")
            return False
            
        except Exception as e:
            self.error_tracker.record_error(
                'HEALTH_CHECK_ERROR',
                f"Unexpected error: {str(e)}",
                {
                    'url': f"{self.base_url}/api/tags",
                    'error_type': type(e).__name__,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
            )
            logger.error(f"[LLM Health] 💥 Unexpected error: {e}")
            logger.error(f"[LLM Health] Traceback:\n{traceback.format_exc()}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics including error statistics"""
        cache_hit_rate = 0.0
        if self.metrics['total_queries'] > 0:
            cache_hit_rate = (self.metrics['cache_hits'] / self.metrics['total_queries']) * 100
        
        success_rate = 0.0
        if self.metrics['total_queries'] > 0:
            success_rate = (self.metrics['successful_queries'] / self.metrics['total_queries']) * 100
        
        error_summary = self.error_tracker.get_error_summary()
        
        return {
            'total_queries': self.metrics['total_queries'],
            'successful_queries': self.metrics['successful_queries'],
            'failed_queries': self.metrics['failed_queries'],
            'success_rate': success_rate,
            'cache_hits': self.metrics['cache_hits'],
            'cache_hit_rate': cache_hit_rate,
            'average_latency': self.metrics['average_latency'],
            'current_model': self.fallback_models[self.current_model_index],
            'cache_size': len(self.response_cache),
            'conversation_history_length': len(self.conversation_history),
            'error_statistics': {
                'total_errors': error_summary['total_errors'],
                'error_counts': error_summary['error_counts'],
                'last_error_type': error_summary['last_error']['type'] if error_summary['last_error'] else None,
                'last_error_time': error_summary['last_error']['timestamp'] if error_summary['last_error'] else None
            },
            'health_status': self.check_health()
        }
    
    def get_error_report(self) -> str:
        """Generate a detailed error report for debugging"""
        error_summary = self.error_tracker.get_error_summary()
        report_lines = [
            "=" * 80,
            "LLM ERROR DEBUG REPORT",
            "=" * 80,
            f"Total Errors: {error_summary['total_errors']}",
            f"Error Types: {json.dumps(error_summary['error_counts'], indent=2)}",
            ""
        ]
        
        if error_summary['last_error']:
            last_err = error_summary['last_error']
            report_lines.extend([
                "LAST ERROR:",
                f"  Type: {last_err['type']}",
                f"  Message: {last_err['message']}",
                f"  Timestamp: {time.ctime(last_err['timestamp'])}",
                f"  Context: {json.dumps(last_err['context'], indent=4)}",
                ""
            ])
        
        if error_summary['recent_errors']:
            report_lines.extend([
                "RECENT ERRORS (last 5):",
                ""
            ])
            for i, err in enumerate(error_summary['recent_errors'][-5:], 1):
                report_lines.extend([
                    f"Error #{i}:",
                    f"  Type: {err['type']}",
                    f"  Message: {err['message']}",
                    f"  Time: {time.ctime(err['timestamp'])}",
                    ""
                ])
        
        report_lines.extend([
            "=" * 80,
            f"Ollama Base URL: {self.base_url}",
            f"Current Model: {self.fallback_models[self.current_model_index]}",
            f"Health Status: {'✅ Healthy' if self.check_health() else '❌ Unhealthy'}",
            "=" * 80
        ])
        
        return "\n".join(report_lines)
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'cache_hits': 0,
            'average_latency': 0.0,
            'latencies': deque(maxlen=100),
            'last_query_time': 0.0
        }
        self.response_cache.clear()
        self.conversation_history.clear()
        self.error_tracker.error_history.clear()
        self.error_tracker.error_counts.clear()
        self.error_tracker.last_error = None
        logger.info("LLM metrics, cache, and error history reset")
    
    def enable_debug_mode(self, enabled: bool = True):
        """Enable or disable detailed debug logging"""
        self.debug_mode = enabled
        logger.info(f"LLM debug mode: {'ENABLED' if enabled else 'DISABLED'}")
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics for troubleshooting"""
        error_summary = self.error_tracker.get_error_summary()
        health_status = self.check_health()
        
        diagnostics = {
            'service_status': {
                'base_url': self.base_url,
                'health_check': health_status,
                'current_model': self.fallback_models[self.current_model_index],
                'available_models': self.fallback_models,
                'timeout': config.OLLAMA_TIMEOUT
            },
            'performance': {
                'total_queries': self.metrics['total_queries'],
                'success_rate': (self.metrics['successful_queries'] / max(1, self.metrics['total_queries'])) * 100,
                'average_latency': self.metrics['average_latency'],
                'cache_hit_rate': (self.metrics['cache_hits'] / max(1, self.metrics['total_queries'])) * 100
            },
            'errors': {
                'total_errors': error_summary['total_errors'],
                'error_breakdown': error_summary['error_counts'],
                'last_error': {
                    'type': error_summary['last_error']['type'] if error_summary['last_error'] else None,
                    'message': error_summary['last_error']['message'] if error_summary['last_error'] else None,
                    'timestamp': error_summary['last_error']['timestamp'] if error_summary['last_error'] else None
                }
            },
            'configuration': {
                'debug_mode': self.debug_mode,
                'max_retries': self.max_retries,
                'retry_delays': self.retry_delays,
                'cache_ttl': self.cache_ttl,
                'min_query_interval': self.min_query_interval
            },
            'recommendations': self._generate_recommendations(error_summary, health_status)
        }
        
        return diagnostics
    
    def _generate_recommendations(self, error_summary: Dict[str, Any], health_status: bool) -> List[str]:
        """Generate troubleshooting recommendations based on current state"""
        recommendations = []
        
        if not health_status:
            recommendations.append("⚠️ Ollama service appears to be down. Start it with: ollama serve")
            recommendations.append(f"⚠️ Verify Ollama is accessible at: {self.base_url}")
        
        if error_summary['total_errors'] > 0:
            error_counts = error_summary['error_counts']
            
            if error_counts.get('CONNECTION_ERROR', 0) > 0:
                recommendations.append("🔌 Connection errors detected. Check network connectivity and Ollama service status.")
            
            if error_counts.get('TIMEOUT', 0) > 0:
                recommendations.append(f"⏱️ Timeout errors detected. Consider increasing OLLAMA_TIMEOUT (current: {config.OLLAMA_TIMEOUT}s)")
            
            if error_counts.get('PARSE_ERROR', 0) > 0:
                recommendations.append("📝 Parse errors detected. LLM may be returning invalid JSON. Check model output format.")
            
            if error_counts.get('HTTP_ERROR', 0) > 0:
                recommendations.append("🌐 HTTP errors detected. Check Ollama service logs for details.")
        
        success_rate = (self.metrics['successful_queries'] / max(1, self.metrics['total_queries'])) * 100
        if success_rate < 50 and self.metrics['total_queries'] > 10:
            recommendations.append(f"📉 Low success rate ({success_rate:.1f}%). Review error breakdown for patterns.")
        
        if not recommendations:
            recommendations.append("✅ No issues detected. System appears healthy.")
        
        return recommendations

