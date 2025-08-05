from models import User, db
from werkzeug.security import generate_password_hash

# Create a new user
new_admin = User(
    username="Admin User",
    email="success@example.com",
    password_hash=generate_password_hash("yourpassword"),
    
)
db.session.add(new_admin)
db.session.commit()
print("Admin created!")
