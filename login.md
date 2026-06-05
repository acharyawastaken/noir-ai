# Noir RAG Engine — Login Credentials

To access the secured endpoints and use the chat/upload functionalities in the `noir` RAG interface, use one of the following login credentials.

## 🔑 Available Credentials

### 1. Default Administrator
* **Username:** `admin`
* **Password:** `admin`
* **Role:** Administrator (Default)

### 2. General User Session (Self-Registration Demo)
* **Username:** *Any username of your choice* (e.g., `sameer`, `guest`, `analyst`)
* **Password:** `password`
* **Role:** General User (Each unique username establishes an independent turn-by-turn chat history session)

---

## 🔒 JWT Authentication Details

The system generates a secure JSON Web Token (JWT) signed with HMAC-SHA256 upon successful login.

* **Key:** `noir-super-secret-key-12345`
* **Algorithm:** `HS256`
* **Token Lifespan:** 24 Hours
* **Authorization Header:** `Authorization: Bearer <token>`
