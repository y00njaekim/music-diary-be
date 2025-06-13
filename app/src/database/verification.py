import os
import jwt
import requests

from flask import request, jsonify

from functools import wraps

def verify_jwt(f):
	SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
	"""JWT 토큰을 검증하는 데코레이터 (대칭키 방식)"""

	@wraps(f)
	def decorated_function(*args, **kwargs):
		auth_header = request.headers.get("Authorization")
		if not auth_header:
			return jsonify({"error": "인증 헤더가 없습니다"}), 401

		if not SUPABASE_JWT_SECRET:
			return jsonify({"error": "서버 설정 오류: JWT 시크릿이 없습니다."}), 500

		try:
			if not auth_header.startswith("Bearer "):
				return jsonify({"error": "잘못된 인증 헤더 형식입니다"}), 401
			token = auth_header.split(" ")[1]
			payload = jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"], audience="authenticated")
			user_id = payload.get("sub")  # user.id
			email = payload.get("email", "")  # user.email
			user_metadata = payload.get("user_metadata", {})
			name = user_metadata.get("name", email.split("@")[0] if email else "User")
			app_metadata = payload.get("app_metadata", {})
			request.jwt_user = {"id": user_id, "email": email, "name": name, "user_metadata": user_metadata,
								"app_metadata": app_metadata}
			return f(*args, **kwargs)
		except jwt.ExpiredSignatureError:
			return jsonify({"error": "토큰이 만료되었습니다"}), 401
		except jwt.InvalidTokenError as e:
			return jsonify({"error": "유효하지 않은 토큰입니다"}), 401
		except Exception as e:
			return jsonify({"error": f"토큰 검증 중 오류: {str(e)}"}), 500

	return decorated_function