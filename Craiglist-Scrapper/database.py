"""
Database connection and operations for storing conversations.
"""

import os
import logging
from typing import List, Dict, Optional
from datetime import datetime
from decimal import Decimal
import mysql.connector
from mysql.connector import Error, pooling

logger = logging.getLogger(__name__)

# Database configuration from environment variables
# IMPORTANT: Never hardcode credentials. Always use environment variables.
DB_HOST = os.getenv('DB_HOST')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME')
DB_PORT = int(os.getenv('DB_PORT', '3306'))

# Connection pool configuration
pool_config = {
    'pool_name': 'conversation_pool',
    'pool_size': 5,
    'pool_reset_session': True,
    'host': DB_HOST,
    'user': DB_USER,
    'password': DB_PASSWORD,
    'database': DB_NAME,
    'port': DB_PORT,
    'autocommit': True,
    'charset': 'utf8mb4',
    'collation': 'utf8mb4_unicode_ci',
}

# Global connection pool
connection_pool: Optional[pooling.MySQLConnectionPool] = None


def get_connection_pool():
    """
    Get or create the database connection pool.
    
    Returns:
        MySQLConnectionPool: The connection pool instance
    """
    global connection_pool
    if connection_pool is None:
        try:
            connection_pool = pooling.MySQLConnectionPool(**pool_config)
            logger.info("Database connection pool created successfully")
        except Error as e:
            logger.error(f"Error creating connection pool: {e}")
            raise
    return connection_pool


def get_db_connection():
    """
    Get a database connection from the pool.
    
    Returns:
        MySQLConnection: A database connection
    """
    try:
        pool = get_connection_pool()
        conn = pool.get_connection()
        # Ensure connection is properly configured
        conn.autocommit = True
        return conn
    except Error as e:
        logger.error(f"Error getting database connection: {e}")
        raise


def init_database():
    """
    Initialize database tables if they don't exist.
    Creates conversations table with proper schema.
    Also checks if table exists and has correct schema, adds missing columns if needed.
    """
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SHOW TABLES LIKE 'conversations'")
        table_exists = cursor.fetchone()
        
        if table_exists:
            # Check if table has correct columns
            cursor.execute("DESCRIBE conversations")
            columns = [row[0] for row in cursor.fetchall()]
            
            required_columns = {
                'id': 'INT AUTO_INCREMENT PRIMARY KEY',
                'thread_id': 'VARCHAR(255) NOT NULL',
                'phone_number': 'VARCHAR(20) NOT NULL',
                'message_type': "ENUM('inbound', 'outbound') NOT NULL",
                'role': "ENUM('user', 'assistant') NOT NULL",
                'content': 'TEXT NOT NULL',
                'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
            }
            
            missing_columns = {col: defn for col, defn in required_columns.items() if col not in columns}
            
            if missing_columns:
                logger.warning(f"Table exists but missing columns: {list(missing_columns.keys())}. Adding missing columns...")
                
                # Add missing columns one by one
                for col_name, col_def in missing_columns.items():
                    try:
                        if col_name == 'id' and 'id' not in columns:
                            # If id is missing, we need to handle it specially
                            logger.warning("Cannot add PRIMARY KEY column 'id' to existing table. Please recreate table manually.")
                            continue
                        
                        alter_query = f"ALTER TABLE conversations ADD COLUMN {col_name} {col_def}"
                        cursor.execute(alter_query)
                        logger.info(f"Added column '{col_name}' to conversations table")
                    except Error as e:
                        logger.warning(f"Could not add column '{col_name}': {e}. It may already exist or have a different definition.")
                
                # Add indexes if they don't exist
                try:
                    cursor.execute("SHOW INDEX FROM conversations WHERE Key_name = 'idx_thread_id'")
                    if not cursor.fetchone():
                        cursor.execute("CREATE INDEX idx_thread_id ON conversations(thread_id)")
                        logger.info("Created index idx_thread_id")
                except Error:
                    pass
                
                try:
                    cursor.execute("SHOW INDEX FROM conversations WHERE Key_name = 'idx_phone_number'")
                    if not cursor.fetchone():
                        cursor.execute("CREATE INDEX idx_phone_number ON conversations(phone_number)")
                        logger.info("Created index idx_phone_number")
                except Error:
                    pass
                
                try:
                    cursor.execute("SHOW INDEX FROM conversations WHERE Key_name = 'idx_created_at'")
                    if not cursor.fetchone():
                        cursor.execute("CREATE INDEX idx_created_at ON conversations(created_at)")
                        logger.info("Created index idx_created_at")
                except Error:
                    pass
                
                conn.commit()
                logger.info("Database table 'conversations' updated with missing columns")
            else:
                logger.info("Database table 'conversations' already exists with correct schema")
        else:
            # Create conversations table from scratch
            create_table_query = """
            CREATE TABLE conversations (
                id INT AUTO_INCREMENT PRIMARY KEY,
                thread_id VARCHAR(255) NOT NULL,
                phone_number VARCHAR(20) NOT NULL,
                message_type ENUM('inbound', 'outbound') NOT NULL,
                role ENUM('user', 'assistant') NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_thread_id (thread_id),
                INDEX idx_phone_number (phone_number),
                INDEX idx_created_at (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """
            
            cursor.execute(create_table_query)
            conn.commit()
            logger.info("Database table 'conversations' created successfully")
        
    except Error as e:
        logger.error(f"Error initializing database: {e}")
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def save_message(thread_id: str, phone_number: str, message_type: str, role: str, content: str) -> bool:
    """
    Save a message to the database.
    
    Args:
        thread_id: Unique thread identifier
        phone_number: Phone number of the participant
        message_type: 'inbound' or 'outbound'
        role: 'user' or 'assistant'
        content: Message content
    
    Returns:
        bool: True if message saved successfully, False otherwise
    """
    import uuid as uuid_lib
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check which columns exist in the table and their types
        cursor.execute("DESCRIBE conversations")
        column_info = {row[0]: row[1] for row in cursor.fetchall()}
        columns = list(column_info.keys())
        has_uuid = 'uuid' in columns
        has_user_id = 'user_id' in columns
        has_agent_id = 'agent_id' in columns
        
        # Determine user_id value based on column type and foreign key constraint
        user_id_value = None
        if has_user_id:
            user_id_type = column_info.get('user_id', '').upper()
            # Check if user_id column allows NULL
            cursor.execute("""
                SELECT IS_NULLABLE 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = DATABASE() 
                AND TABLE_NAME = 'conversations' 
                AND COLUMN_NAME = 'user_id'
            """)
            nullable_result = cursor.fetchone()
            is_nullable = nullable_result and nullable_result[0] == 'YES'
            
            if 'INT' in user_id_type:
                # user_id is an integer foreign key - need to find or create user
                try:
                    # First, check if users table exists and what columns it has
                    cursor.execute("SHOW TABLES LIKE 'users'")
                    users_table_exists = cursor.fetchone()
                    
                    if users_table_exists:
                        # Get columns from users table
                        cursor.execute("DESCRIBE users")
                        users_columns = [row[0] for row in cursor.fetchall()]
                        
                        # Try to find user - check various possible column names
                        user_result = None
                        if 'phone_number' in users_columns:
                            cursor.execute("SELECT id FROM users WHERE phone_number = %s LIMIT 1", (phone_number,))
                            user_result = cursor.fetchone()
                        elif 'phone' in users_columns:
                            cursor.execute("SELECT id FROM users WHERE phone = %s LIMIT 1", (phone_number,))
                            user_result = cursor.fetchone()
                        elif 'email' in users_columns:
                            # If no phone column, try to find by email or use first user
                            cursor.execute("SELECT id FROM users LIMIT 1")
                            user_result = cursor.fetchone()
                        
                        if user_result:
                            user_id_value = user_result[0]
                        else:
                            # User doesn't exist - try to create one
                            try:
                                # Build INSERT based on available columns
                                insert_cols = []
                                insert_vals = []
                                
                                if 'phone_number' in users_columns:
                                    insert_cols.append('phone_number')
                                    insert_vals.append(phone_number)
                                elif 'phone' in users_columns:
                                    insert_cols.append('phone')
                                    insert_vals.append(phone_number)
                                
                                if 'created_at' in users_columns:
                                    insert_cols.append('created_at')
                                    insert_vals.append('NOW()')
                                
                                if insert_cols:
                                    cols_str = ', '.join(insert_cols)
                                    vals_str = ', '.join(['%s' if v != 'NOW()' else 'NOW()' for v in insert_vals])
                                    insert_vals_clean = [v for v in insert_vals if v != 'NOW()']
                                    
                                    cursor.execute(f"""
                                        INSERT INTO users ({cols_str}) 
                                        VALUES ({vals_str})
                                    """, tuple(insert_vals_clean))
                                    conn.commit()
                                    user_id_value = cursor.lastrowid
                                    logger.info(f"Created new user with id {user_id_value} for phone {phone_number}")
                                else:
                                    # Can't create user without any insertable columns
                                    if is_nullable:
                                        user_id_value = None
                                        logger.warning(f"Could not create user - no suitable columns in users table")
                                    else:
                                        # Use first available user ID or default
                                        cursor.execute("SELECT id FROM users LIMIT 1")
                                        first_user = cursor.fetchone()
                                        user_id_value = first_user[0] if first_user else 1
                                        logger.warning(f"Using existing user_id={user_id_value} as fallback")
                            except Error as create_error:
                                # If we can't create user and column is nullable, use NULL
                                if is_nullable:
                                    user_id_value = None
                                    logger.warning(f"Could not create user for {phone_number}, using NULL for user_id: {create_error}")
                                else:
                                    # Try to get any existing user ID
                                    try:
                                        cursor.execute("SELECT id FROM users LIMIT 1")
                                        first_user = cursor.fetchone()
                                        user_id_value = first_user[0] if first_user else 1
                                        logger.warning(f"Using existing user_id={user_id_value} as fallback")
                                    except:
                                        user_id_value = 1
                                        logger.warning(f"Using default user_id=1 as final fallback")
                    else:
                        # Users table doesn't exist
                        if is_nullable:
                            user_id_value = None
                            logger.warning("Users table doesn't exist, using NULL for user_id")
                        else:
                            user_id_value = 1
                            logger.warning("Users table doesn't exist, using default user_id=1")
                            
                except Error as lookup_error:
                    # If users table doesn't exist or lookup fails
                    if is_nullable:
                        user_id_value = None
                        logger.warning(f"Could not lookup user for {phone_number}, using NULL for user_id: {lookup_error}")
                    else:
                        # If not nullable and we can't find/create user, use a default (1) as fallback
                        user_id_value = 1
                        logger.warning(f"Could not lookup/create user, using default user_id=1: {lookup_error}")
            elif 'VARCHAR' in user_id_type or 'TEXT' in user_id_type or 'CHAR' in user_id_type:
                # If user_id is a string, use phone_number
                user_id_value = phone_number
            else:
                # Default: try to use phone_number as string
                user_id_value = phone_number
        
        # Determine agent_id value based on column type
        agent_id_value = None
        if has_agent_id:
            agent_id_type = column_info.get('agent_id', '').upper()
            if 'INT' in agent_id_type:
                # If agent_id is an integer, use a default agent ID (e.g., 1 for AI assistant)
                agent_id_value = 1  # Default AI agent ID
            elif 'VARCHAR' in agent_id_type or 'TEXT' in agent_id_type or 'CHAR' in agent_id_type:
                # If agent_id is a string, use a default agent identifier
                agent_id_value = 'ai_assistant'  # Default AI agent identifier
            else:
                # Default: use 1 as integer
                agent_id_value = 1
        
        # Build column list and values list dynamically
        insert_columns = []
        insert_values = []
        
        if has_uuid:
            insert_columns.append('uuid')
            insert_values.append(str(uuid_lib.uuid4()))
        
        if has_user_id:
            insert_columns.append('user_id')
            insert_values.append(user_id_value)
        
        if has_agent_id:
            insert_columns.append('agent_id')
            insert_values.append(agent_id_value)
        
        # Always include these columns
        insert_columns.extend(['thread_id', 'phone_number', 'message_type', 'role', 'content'])
        insert_values.extend([thread_id, phone_number, message_type, role, content])
        
        # Build the INSERT query
        columns_str = ', '.join(insert_columns)
        placeholders = ', '.join(['%s'] * len(insert_values))
        insert_query = f"""
        INSERT INTO conversations ({columns_str})
        VALUES ({placeholders})
        """
        
        cursor.execute(insert_query, tuple(insert_values))
        
        conn.commit()
        
        logger.info(f"Message saved successfully: thread_id={thread_id}, phone={phone_number}, type={message_type}, role={role}")
        return True
        
    except Error as e:
        logger.error(f"Error saving message to database: {e}")
        return False
    finally:
        # Always close cursor and connection to release back to pool
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_conversations_by_thread(thread_id: str) -> List[Dict]:
    """
    Get all messages for a specific thread.
    
    Args:
        thread_id: Thread identifier
    
    Returns:
        List of message dictionaries ordered by creation time
    """
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        select_query = """
        SELECT id, thread_id, phone_number, message_type, role, content, created_at
        FROM conversations
        WHERE thread_id = %s
        ORDER BY created_at ASC
        """
        
        cursor.execute(select_query, (thread_id,))
        results = cursor.fetchall()
        
        # Convert datetime to ISO format string and Decimal to int/float
        for result in results:
            # Convert Decimal values to int/float for JSON serialization
            for key, value in result.items():
                if isinstance(value, Decimal):
                    result[key] = int(value) if value % 1 == 0 else float(value)
            
            # Convert datetime to ISO format string
            if result.get('created_at'):
                if hasattr(result['created_at'], 'isoformat'):
                    result['created_at'] = result['created_at'].isoformat()
        
        return results
        
    except Error as e:
        logger.error(f"Error fetching conversations by thread: {e}")
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_all_threads() -> List[Dict]:
    """
    Get all unique threads with summary information.
    
    Returns:
        List of thread summary dictionaries
    """
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # First, check if table has any data
        cursor.execute("SELECT COUNT(*) as total FROM conversations")
        count_result = cursor.fetchone()
        total_count = count_result.get('total', 0) if count_result else 0
        logger.info(f"Total messages in conversations table: {total_count}")
        
        if total_count == 0:
            logger.warning("No messages found in conversations table")
            return []
        
        # Get column names to build query dynamically
        # Note: When using dictionary=True, DESCRIBE returns dicts with 'Field' as the key
        cursor.execute("DESCRIBE conversations")
        describe_results = cursor.fetchall()
        # Handle both dict and tuple results
        if describe_results and isinstance(describe_results[0], dict):
            table_columns = [row.get('Field', row.get('field', '')) for row in describe_results]
        else:
            table_columns = [row[0] for row in describe_results]
        logger.debug(f"Conversations table columns: {table_columns}")
        
        # Build query with available columns
        select_query = """
        SELECT 
            thread_id,
            phone_number,
            COUNT(*) as message_count,
            MIN(created_at) as first_message,
            MAX(created_at) as last_message,
            SUM(CASE WHEN message_type = 'inbound' THEN 1 ELSE 0 END) as inbound_count,
            SUM(CASE WHEN message_type = 'outbound' THEN 1 ELSE 0 END) as outbound_count
        FROM conversations
        GROUP BY thread_id, phone_number
        ORDER BY last_message DESC
        """
        
        cursor.execute(select_query)
        results = cursor.fetchall()
        
        logger.info(f"Found {len(results)} conversation threads")
        
        # Convert datetime to ISO format string and Decimal to int/float
        for result in results:
            # Convert Decimal values to int/float for JSON serialization
            for key, value in result.items():
                if isinstance(value, Decimal):
                    result[key] = int(value) if value % 1 == 0 else float(value)
            
            # Convert datetime to ISO format string
            if result.get('first_message'):
                if hasattr(result['first_message'], 'isoformat'):
                    result['first_message'] = result['first_message'].isoformat()
            if result.get('last_message'):
                if hasattr(result['last_message'], 'isoformat'):
                    result['last_message'] = result['last_message'].isoformat()
        
        return results
        
    except Error as e:
        logger.error(f"Error fetching all threads: {e}", exc_info=True)
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_messages_by_type(thread_id: str, message_type: str) -> List[Dict]:
    """
    Get messages filtered by type (inbound/outbound) for a thread.
    
    Args:
        thread_id: Thread identifier
        message_type: 'inbound' or 'outbound'
    
    Returns:
        List of message dictionaries
    """
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        select_query = """
        SELECT id, thread_id, phone_number, message_type, role, content, created_at
        FROM conversations
        WHERE thread_id = %s AND message_type = %s
        ORDER BY created_at ASC
        """
        
        cursor.execute(select_query, (thread_id, message_type))
        results = cursor.fetchall()
        
        # Convert datetime to ISO format string and Decimal to int/float
        for result in results:
            # Convert Decimal values to int/float for JSON serialization
            for key, value in result.items():
                if isinstance(value, Decimal):
                    result[key] = int(value) if value % 1 == 0 else float(value)
            
            # Convert datetime to ISO format string
            if result.get('created_at'):
                if hasattr(result['created_at'], 'isoformat'):
                    result['created_at'] = result['created_at'].isoformat()
        
        return results
        
    except Error as e:
        logger.error(f"Error fetching messages by type: {e}")
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def phone_number_exists(phone_number: str) -> bool:
    """
    Check if a phone number already exists in the database.
    
    This function checks if there are any conversations (messages) associated with
    the given phone number. If any messages exist for this phone number, it means
    the number has been contacted before.
    
    Args:
        phone_number: Phone number to check (can be in any format, will be normalized for comparison)
    
    Returns:
        bool: True if phone number exists in database, False otherwise
    """
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Normalize phone number for comparison - remove common formatting characters
        # This handles variations like +1-555-123-4567, (555) 123-4567, 5551234567, etc.
        normalized_phone = ''.join(c for c in phone_number if c.isdigit() or c == '+')
        
        # Check if phone number exists in database
        # Try exact match first
        select_query = """
        SELECT COUNT(*) as count
        FROM conversations
        WHERE phone_number = %s
        LIMIT 1
        """
        
        cursor.execute(select_query, (phone_number,))
        result = cursor.fetchone()
        
        if result and result[0] > 0:
            logger.debug(f"Phone number {phone_number} found in database (exact match)")
            return True
        
        # If exact match fails, try normalized comparison
        # This handles cases where phone numbers might be stored in different formats
        # Get all phone numbers and compare normalized versions
        cursor.execute("SELECT DISTINCT phone_number FROM conversations")
        all_phones = cursor.fetchall()
        
        for db_phone_tuple in all_phones:
            db_phone = db_phone_tuple[0] if db_phone_tuple else ''
            if db_phone:
                # Normalize database phone number
                normalized_db_phone = ''.join(c for c in str(db_phone) if c.isdigit() or c == '+')
                # Compare last 10 digits (removing country code variations)
                phone_digits = normalized_phone[-10:] if len(normalized_phone) >= 10 else normalized_phone
                db_phone_digits = normalized_db_phone[-10:] if len(normalized_db_phone) >= 10 else normalized_db_phone
                
                if phone_digits == db_phone_digits and len(phone_digits) == 10:
                    logger.debug(f"Phone number {phone_number} found in database (normalized match with {db_phone})")
                    return True
        
        logger.debug(f"Phone number {phone_number} not found in database")
        return False
        
    except Error as e:
        logger.error(f"Error checking if phone number exists: {e}")
        # On error, return False to allow sending (fail-safe approach)
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
