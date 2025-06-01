# Music Diary Backend

음악 일기 백엔드 API 서버입니다.

## 실행 방법

### Docker Compose를 사용한 실행

#### HTTP (80번 포트) 사용
```bash
docker compose up -d
```

#### HTTPS (443번 포트) 사용
1. SSL 인증서를 `ssl/` 디렉토리에 준비합니다:
   - `ssl/cert.pem`: SSL 인증서
   - `ssl/key.pem`: SSL 개인 키

2. HTTPS용 Docker Compose 실행:
```bash
docker compose -f docker-compose.https.yml up -d
```

### 포트 설정
- **HTTP**: 호스트의 80번 포트 → 컨테이너의 5000번 포트
- **HTTPS**: 호스트의 443번 포트 → Nginx → 컨테이너의 5000번 포트

### 서비스 중지
```bash
docker compose down
```

## 개발 환경
- Python 3.12
- Flask
- Poetry (의존성 관리)
