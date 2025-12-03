"""
Integrated Vehicle Lead Generation Pipeline
Combines all functionalities: scraping, phone extraction, MMR pricing, lead qualification, and outreach
"""

import os
import time
import csv
import re
import json
import logging
import pandas as pd
import requests
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv

# Selenium imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager

# Twilio and OpenAI imports
from twilio.rest import Client
from openai import OpenAI

# Database imports for saving SMS messages
try:
    from database import save_message
except ImportError:
    # If database module is not available, create a no-op function
    def save_message(*args, **kwargs):
        logger.warning("Database module not available - SMS messages will not be saved to database")
        return False

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class PipelineConfig:
    """Centralized configuration for the entire pipeline"""
    
    # Craigslist Settings
    TARGET_URL = "https://losangeles.craigslist.org/search/cta?bundleDuplicates=1&purveyor=owner#search=2~gallery~0"
    HEADLESS_MODE = False
    
    # Manheim API
    MANHEIM_ENDPOINT = os.getenv("MANHEIM_ENDPOINT")
    MANHEIM_AUTH = os.getenv("MANHEIM_AUTH")
    MANHEIM_API_BASE_URL = os.getenv("MANHEIM_API_BASE_URL")
    
    # Salesforce API
    SALESFORCE_API_URL = "https://iframe-backend.vercel.app/api/submit-chatbot-form"
    
    # Twilio Settings
    TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
    TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
    TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
    
    # OpenAI Settings
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    INITIAL_MESSAGE = os.getenv('INITIAL_MESSAGE', 
        "Hello this is Carly from Car Trackers, I'm reaching out for your vehicle that you have posted on Craigslist for sale. Could you please share the VIN number of your vehicle?")
    
    # Output Files
    SCRAPED_DATA_FILE = f"scraped_vehicles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    QUALIFIED_LEADS_FILE = f"qualified_leads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Processing Settings
    API_DELAY = 0.5  # Delay between API calls
    SCRAPING_DELAY = 2  # Delay between page loads


# =============================================================================
# STEP 1: WEB SCRAPING
# =============================================================================

class CraigslistScraper:
    """Scrapes Craigslist vehicle listings"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.driver = None
        self.wait = None
        self._clicked_show_contact_recently = False
        self._setup_chrome_driver()
    
    def _setup_chrome_driver(self):
        """Setup Chrome WebDriver"""
        logger.info("Setting up Chrome WebDriver...")
        
        chrome_options = Options()
        if self.config.HEADLESS_MODE:
            chrome_options.add_argument("--headless=new")
        
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.wait = WebDriverWait(self.driver, 15)
        
        logger.info("Chrome WebDriver initialized successfully")
    
    def scrape_listings(self) -> List[Dict]:
        """
        Scrape all vehicle listings from Craigslist
        Returns list of vehicle dictionaries
        """
        try:
            logger.info("=" * 70)
            logger.info("STARTING CRAIGSLIST SCRAPING")
            logger.info("=" * 70)
            
            # Navigate to target URL
            logger.info(f"Navigating to: {self.config.TARGET_URL}")
            self.driver.get(self.config.TARGET_URL)
            time.sleep(40)  # Wait for page load
            
            # Apply filters
            self._click_owner_filter()
            self._click_bundle_duplicates()
            time.sleep(5)
            
            # Get all gallery cards
            gallery_cards = self._get_gallery_cards()
            total_cards = len(gallery_cards)
            logger.info(f"Found {total_cards} vehicle listings to scrape")
            
            # Scrape each listing
            vehicles = []
            for index in range(total_cards):
                try:
                    logger.info(f"Processing listing {index + 1}/{total_cards}")
                    
                    # Re-fetch cards to avoid stale references
                    current_cards = self._get_gallery_cards()
                    if index >= len(current_cards):
                        break
                    
                    card = current_cards[index]
                    
                    # Get listing URL before clicking
                    listing_url = self._get_listing_url(card)
                    
                    # Click and scrape
                    if self._click_gallery_card(card):
                        vehicle_data = self._scrape_vehicle_details()
                        vehicle_data['listing_url'] = listing_url
                        vehicle_data['scraped_at'] = datetime.now().isoformat()
                        vehicles.append(vehicle_data)
                        
                        self._go_back_to_gallery()
                    
                    time.sleep(self.config.SCRAPING_DELAY)
                    
                except Exception as e:
                    logger.error(f"Error processing listing {index + 1}: {str(e)}")
                    continue
            
            logger.info(f"Successfully scraped {len(vehicles)} vehicles")
            return vehicles
            
        except Exception as e:
            logger.error(f"Error during scraping: {str(e)}")
            return []
        finally:
            self.close()
    
    def _click_owner_filter(self):
        """Click owner filter button"""
        try:
            owner_button = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, "//div[@class='cl-segmented-selector']//button[span[text()='owner']]"))
            )
            owner_button.click()
            logger.info("Owner filter applied")
        except Exception as e:
            logger.warning(f"Could not apply owner filter: {str(e)}")
    
    def _click_bundle_duplicates(self):
        """Click bundle duplicates checkbox"""
        try:
            bundle_checkbox = self.wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "input[name='bundleDuplicates']"))
            )
            bundle_checkbox.click()
            logger.info("Bundle duplicates enabled")
        except Exception as e:
            logger.warning(f"Could not enable bundle duplicates: {str(e)}")
    
    def _get_gallery_cards(self):
        """Get all gallery cards"""
        try:
            return self.wait.until(
                EC.presence_of_all_elements_located((By.XPATH, "//div[@class='gallery-card']"))
            )
        except TimeoutException:
            return []
    
    def _get_listing_url(self, card_element) -> str:
        """Extract listing URL from gallery card"""
        try:
            link = card_element.find_element(By.TAG_NAME, "a")
            return link.get_attribute("href")
        except:
            return "URL not found"
    
    def _click_gallery_card(self, card_element) -> bool:
        """Click on a gallery card"""
        try:
            self.driver.execute_script("arguments[0].scrollIntoView(true);", card_element)
            time.sleep(1)
            
            try:
                card_element.click()
            except:
                self.driver.execute_script("arguments[0].click();", card_element)
            
            self.wait.until(
                EC.presence_of_element_located((By.XPATH, "//span[@id='titletextonly']"))
            )
            return True
        except Exception as e:
            logger.error(f"Error clicking card: {str(e)}")
            return False
    
    def _scrape_vehicle_details(self) -> Dict:
        """Scrape details from vehicle page"""
        # Click show-contact button to reveal phone number
        self._click_show_contact()
        
        # Wait a bit for any dynamic content to load
        time.sleep(1)
        
        vehicle = {
            'title': self._safe_extract("//span[@id='titletextonly']"),
            'price': self._safe_extract("//span[@class='price']"),
            'mileage': self._safe_extract("//div[@class='attr auto_miles']/span[@class='valu']"),
            'vin': self._safe_extract("//div[@class='attr auto_vin']/span[@class='valu']"),
            'vehicle_details': self._safe_extract("//section[@id='postingbody']")
        }
        
        # Try to extract phone number from multiple possible locations
        phone_number = self._extract_phone_from_page()
        if phone_number:
            vehicle['vehicle_details'] += f"\n\nContact: {phone_number}"
        
        logger.info(f"Scraped: {vehicle['title'][:50]}...")
        return vehicle
    
    def _extract_phone_from_page(self) -> str:
        """Try to extract phone number from various page elements"""
        phone_selectors = [
            "//div[@class='reply-info']//a[contains(@href, 'tel:')]",
            "//a[contains(@href, 'tel:')]",
            "//div[@class='reply-info']",
            "//div[contains(@class, 'phone')]",
            "//span[contains(@class, 'phone')]"
        ]
        
        for selector in phone_selectors:
            try:
                element = self.driver.find_element(By.XPATH, selector)
                text = element.text.strip()
                if text and any(char.isdigit() for char in text):
                    logger.info(f"Found potential phone: {text}")
                    return text
                
                # Check href attribute for tel: links
                href = element.get_attribute('href')
                if href and 'tel:' in href:
                    phone = href.replace('tel:', '').strip()
                    logger.info(f"Found phone in tel link: {phone}")
                    return phone
            except:
                continue
        
        # Last resort: check entire page source for phone patterns
        try:
            page_source = self.driver.page_source
            phone_pattern = r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'
            matches = re.findall(phone_pattern, page_source)
            if matches:
                logger.info(f"Found phone in page source: {matches[0]}")
                return matches[0]
        except:
            pass
        
        return ""
    
    def _click_show_contact(self):
        """Click show-contact button to reveal contact info"""
        try:
            # Try multiple possible selectors for the show contact button
            show_contact_selectors = [
                "//a[@class='show-contact']",
                "//button[contains(@class, 'show-contact')]",
                "//a[contains(text(), 'show contact')]",
                "//button[contains(text(), 'show contact')]"
            ]
            
            for selector in show_contact_selectors:
                try:
                    show_contact_button = self.driver.find_element(By.XPATH, selector)
                    if show_contact_button.is_displayed() and show_contact_button.is_enabled():
                        # Scroll to button
                        self.driver.execute_script("arguments[0].scrollIntoView(true);", show_contact_button)
                        time.sleep(1)
                        
                        # Click first time
                        try:
                            show_contact_button.click()
                        except:
                            self.driver.execute_script("arguments[0].click();", show_contact_button)
                        
                        logger.info("Show-contact button clicked first time")
                        self._clicked_show_contact_recently = True
                        time.sleep(5)
                        
                        # Try clicking second time if button still exists
                        try:
                            show_contact_button.click()
                            logger.info("Show-contact button clicked second time")
                            time.sleep(2)
                        except:
                            logger.info("Second click not needed or button disappeared")
                        
                        return
                except NoSuchElementException:
                    continue
            
            logger.info("No show-contact button found (contact info may already be visible)")
        except Exception as e:
            logger.warning(f"Error clicking show-contact button: {str(e)}")
    
    def _safe_extract(self, xpath: str) -> str:
        """Safely extract text from element"""
        try:
            element = self.driver.find_element(By.XPATH, xpath)
            return element.text.strip()
        except:
            return "Not Found"
    
    def _go_back_to_gallery(self):
        """Navigate back to gallery"""
        try:
            self.driver.back()
            if self._clicked_show_contact_recently:
                time.sleep(0.5)
                self.driver.back()
                self._clicked_show_contact_recently = False
            
            self.wait.until(
                EC.presence_of_all_elements_located((By.XPATH, "//div[@class='gallery-card']"))
            )
        except Exception as e:
            logger.error(f"Error navigating back: {str(e)}")
    
    def close(self):
        """Close browser"""
        if self.driver:
            self.driver.quit()
            logger.info("Browser closed")


# =============================================================================
# STEP 2: PHONE NUMBER EXTRACTION
# =============================================================================

class PhoneExtractor:
    """Extract phone numbers from vehicle details"""
    
    @staticmethod
    def extract_phones(text: str) -> List[str]:
        """
        Extract phone numbers from text
        Returns list of formatted phone numbers
        """
        if not text:
            return []
        
        patterns = [
            r'\b\(?([2-9]\d{2})\)?[-.\s]?([2-9]\d{2})[-.\s]?(\d{4})\b',
            r'\b([2-9]\d{2})[-.\s]?([2-9]\d{2})[-.\s]?(\d{4})\b',
        ]
        
        phone_numbers = []
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                formatted = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
                phone_numbers.append(formatted)
        
        # Standalone 10-digit numbers
        standalone = r'\b([2-9]\d{2})([2-9]\d{2})(\d{4})\b'
        for match in re.finditer(standalone, text):
            formatted = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
            phone_numbers.append(formatted)
        
        # Remove duplicates
        seen = set()
        unique = []
        for phone in phone_numbers:
            cleaned = re.sub(r'[^\d]', '', phone)
            if cleaned not in seen and len(cleaned) == 10:
                seen.add(cleaned)
                unique.append(phone)
        
        return unique
    
    @staticmethod
    def add_phones_to_vehicles(vehicles: List[Dict]) -> List[Dict]:
        """
        Add extracted phone numbers to vehicle dictionaries
        """
        logger.info("=" * 70)
        logger.info("EXTRACTING PHONE NUMBERS")
        logger.info("=" * 70)
        
        for vehicle in vehicles:
            details = vehicle.get('vehicle_details', '')
            phones = PhoneExtractor.extract_phones(details)
            vehicle['phone_numbers'] = phones
            vehicle['phone_count'] = len(phones)
            
            if phones:
                logger.info(f"Found {len(phones)} phone(s) for: {vehicle['title'][:50]}")
        
        total_with_phones = sum(1 for v in vehicles if v['phone_count'] > 0)
        logger.info(f"Total vehicles with phone numbers: {total_with_phones}/{len(vehicles)}")
        
        return vehicles


# =============================================================================
# STEP 3: MMR PRICING
# =============================================================================

class MMRPricingService:
    """Fetch Manheim Market Report pricing data"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._cached_token = None
        self._token_expiry = 0
    
    def authenticate(self) -> str:
        """Authenticate with Manheim API"""
        if self._cached_token and time.time() < self._token_expiry:
            return self._cached_token
        
        headers = {
            "Authorization": f"Basic {self.config.MANHEIM_AUTH}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {"grant_type": "client_credentials"}
        
        response = requests.post(self.config.MANHEIM_ENDPOINT, headers=headers, data=data)
        response.raise_for_status()
        
        token_data = response.json()
        self._cached_token = token_data.get("access_token")
        expires_in = token_data.get("expires_in", 1800)
        self._token_expiry = time.time() + expires_in - 60
        
        logger.info("Manheim API authenticated successfully")
        return self._cached_token
    
    def get_valuation(self, vin: str, mileage: int) -> Optional[float]:
        """Get valuation for a vehicle"""
        try:
            token = self.authenticate()
            headers = {"Authorization": f"Bearer {token}"}
            url = f"{self.config.MANHEIM_API_BASE_URL}/valuations/vin/{vin}?grade=45&odometer={mileage}"
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            items = data.get("items", [])
            if not items:
                return None
            
            return items[0]["adjustedPricing"]["wholesale"]["average"]
        except Exception as e:
            logger.error(f"Error getting valuation for VIN {vin}: {str(e)}")
            return None
    
    def add_mmr_pricing(self, vehicles: List[Dict]) -> List[Dict]:
        """Add MMR pricing to all vehicles"""
        logger.info("=" * 70)
        logger.info("FETCHING MMR PRICING")
        logger.info("=" * 70)
        
        for vehicle in vehicles:
            vin = vehicle.get('vin', '').strip()
            mileage_str = vehicle.get('mileage', '').replace(',', '').strip()
            
            if not vin or vin.lower() == 'not found':
                vehicle['mmr_price'] = None
                continue
            
            try:
                mileage = int(mileage_str)
            except ValueError:
                vehicle['mmr_price'] = None
                continue
            
            logger.info(f"Fetching MMR for: {vehicle['title'][:50]}...")
            mmr_price = self.get_valuation(vin, mileage)
            vehicle['mmr_price'] = mmr_price
            
            if mmr_price:
                logger.info(f"MMR Price: ${mmr_price:,.2f}")
            
            time.sleep(self.config.API_DELAY)
        
        successful = sum(1 for v in vehicles if v.get('mmr_price') is not None)
        logger.info(f"Successfully fetched MMR for {successful}/{len(vehicles)} vehicles")
        
        return vehicles


# =============================================================================
# STEP 4: LEAD QUALIFICATION
# =============================================================================

class LeadQualifier:
    """Qualify leads based on price vs MMR"""
    
    @staticmethod
    def clean_price(price_str: str) -> Optional[float]:
        """Convert price string to float"""
        if not price_str or price_str == "Not Found":
            return None
        
        cleaned = re.sub(r'[^0-9]', '', price_str)
        try:
            return float(cleaned) if cleaned else None
        except:
            return None
    
    @staticmethod
    def qualify_leads(vehicles: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Separate qualified leads from non-leads
        Qualified = Original Price < MMR Price
        """
        logger.info("=" * 70)
        logger.info("QUALIFYING LEADS")
        logger.info("=" * 70)
        
        qualified_leads = []
        non_leads = []
        
        for vehicle in vehicles:
            original_price = LeadQualifier.clean_price(vehicle.get('price', ''))
            mmr_price = vehicle.get('mmr_price')
            
            # Skip if missing critical data
            if original_price is None or mmr_price is None:
                non_leads.append(vehicle)
                continue
            
            vehicle['original_price_clean'] = original_price
            vehicle['price_difference'] = mmr_price - original_price
            # Avoid division by zero
            if mmr_price > 0:
                vehicle['price_difference_pct'] = ((mmr_price - original_price) / mmr_price) * 100
            else:
                vehicle['price_difference_pct'] = 0.0
            
            # Qualify if original price is less than MMR
            if original_price < mmr_price:
                vehicle['is_qualified_lead'] = True
                qualified_leads.append(vehicle)
                logger.info(f"[QUALIFIED LEAD] {vehicle['title'][:50]}")
                logger.info(f"  Price: ${original_price:,.0f} | MMR: ${mmr_price:,.0f} | Savings: ${vehicle['price_difference']:,.0f}")
            else:
                vehicle['is_qualified_lead'] = False
                non_leads.append(vehicle)
        
        logger.info(f"Qualified Leads: {len(qualified_leads)}")
        logger.info(f"Non-Leads: {len(non_leads)}")
        
        return qualified_leads, non_leads


# =============================================================================
# STEP 5: SALESFORCE INTEGRATION
# =============================================================================

class SalesforceIntegration:
    """Send qualified leads to Salesforce"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def send_lead(self, vehicle: Dict) -> bool:
        """Send a single lead to Salesforce"""
        # Parse name from title (fallback to generic if not found)
        first_name, last_name = self._parse_name(vehicle['title'])
        
        # Get primary phone number
        phones = vehicle.get('phone_numbers', [])
        primary_phone = phones[0] if phones else ""
        
        payload = {
            "contact": {
                "first_name": first_name,
                "last_name": last_name,
                "email": "",  # Not available from scraping
                "phone": primary_phone
            },
            "vehicle": {
                "vin": vehicle.get('vin', ''),
                "mileage": vehicle.get('mileage', '').replace(',', '')
            }
        }
        
        try:
            response = requests.post(
                self.config.SALESFORCE_API_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"✓ Successfully sent to Salesforce: {vehicle['title'][:50]}")
                return True
            else:
                # Log detailed error information including response body
                error_details = {
                    'status_code': response.status_code,
                    'response_text': response.text[:500] if response.text else 'No response body',
                    'payload_sent': payload
                }
                logger.error(f"✗ Salesforce error: HTTP {response.status_code}")
                logger.error(f"  Response: {error_details['response_text']}")
                logger.error(f"  Payload: {json.dumps(payload, indent=2)}")
                return False
                
        except requests.exceptions.Timeout:
            logger.error(f"✗ Salesforce timeout: Request took longer than 10 seconds")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"✗ Error sending to Salesforce: {str(e)}")
            logger.error(f"  Exception type: {type(e).__name__}")
            return False
        except Exception as e:
            logger.error(f"✗ Unexpected error sending to Salesforce: {str(e)}")
            logger.error(f"  Exception type: {type(e).__name__}")
            return False
    
    def send_all_leads(self, qualified_leads: List[Dict]) -> Dict:
        """Send all qualified leads to Salesforce"""
        logger.info("=" * 70)
        logger.info("SENDING LEADS TO SALESFORCE")
        logger.info("=" * 70)
        
        results = {
            'sent': 0,
            'failed': 0,
            'skipped': 0
        }
        
        for lead in qualified_leads:
            # Skip if no phone number
            if not lead.get('phone_numbers'):
                logger.warning(f"Skipping (no phone): {lead['title'][:50]}")
                results['skipped'] += 1
                continue
            
            if self.send_lead(lead):
                results['sent'] += 1
            else:
                results['failed'] += 1
            
            time.sleep(self.config.API_DELAY)
        
        logger.info(f"Salesforce Results: {results['sent']} sent, {results['failed']} failed, {results['skipped']} skipped")
        return results
    
    @staticmethod
    def _parse_name(title: str) -> Tuple[str, str]:
        """Parse name from vehicle title"""
        words = title.split()
        if len(words) > 0:
            first_word = words[0].strip("'s").strip("'")
            if first_word and first_word[0].isupper() and not first_word[0].isdigit():
                if "'" in first_word:
                    name_part = first_word.split("'")[0]
                    return name_part, "Owner"
                return first_word, "Owner"
        return "Vehicle", "Owner"


# =============================================================================
# STEP 6: SMS OUTREACH
# =============================================================================

class SMSOutreach:
    """Send SMS messages to qualified leads"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
    
    def format_phone(self, phone: str) -> str:
        """Format phone to E.164"""
        cleaned = ''.join(c for c in phone if c.isdigit() or c == '+')
        if not cleaned.startswith('+'):
            if len(cleaned) == 10:
                cleaned = '+1' + cleaned
            elif len(cleaned) == 11 and cleaned.startswith('1'):
                cleaned = '+' + cleaned
            else:
                cleaned = '+1' + cleaned
        return cleaned
    
    def send_sms(self, to_number: str, message: str) -> bool:
        """
        Send a single SMS and save it to the database.
        
        Args:
            to_number: Phone number to send message to
            message: Message content to send
            
        Returns:
            bool: True if message sent successfully, False otherwise
        """
        try:
            formatted_number = self.format_phone(to_number)
            
            message_obj = self.client.messages.create(
                body=message,
                from_=self.config.TWILIO_PHONE_NUMBER,
                to=formatted_number
            )
            
            logger.info(f"✓ SMS sent to {formatted_number} - SID: {message_obj.sid}")
            
            # Save outbound message to database so it appears in the frontend
            # Use phone number as thread_id for consistency with messaging.py
            thread_id = formatted_number
            try:
                save_message(
                    thread_id=thread_id,
                    phone_number=formatted_number,
                    message_type='outbound',
                    role='assistant',
                    content=message
                )
                logger.debug(f"Saved SMS message to database for {formatted_number}")
            except Exception as db_error:
                # Log but don't fail the SMS send if database save fails
                logger.warning(f"Failed to save SMS to database: {str(db_error)}")
            
            return True
        except Exception as e:
            logger.error(f"✗ Failed to send SMS to {to_number}: {str(e)}")
            return False
    
    def contact_qualified_leads(self, qualified_leads: List[Dict]) -> Dict:
        """Send SMS to all qualified leads"""
        logger.info("=" * 70)
        logger.info("SENDING SMS TO QUALIFIED LEADS")
        logger.info("=" * 70)
        
        results = {
            'sent': 0,
            'failed': 0,
            'no_phone': 0
        }
        
        for lead in qualified_leads:
            phones = lead.get('phone_numbers', [])
            
            if not phones:
                results['no_phone'] += 1
                continue
            
            # Send to primary phone only
            primary_phone = phones[0]
            logger.info(f"Contacting: {lead['title'][:50]}")
            
            if self.send_sms(primary_phone, self.config.INITIAL_MESSAGE):
                results['sent'] += 1
            else:
                results['failed'] += 1
            
            time.sleep(self.config.API_DELAY)
        
        logger.info(f"SMS Results: {results['sent']} sent, {results['failed']} failed, {results['no_phone']} no phone")
        return results


# =============================================================================
# STEP 7: DATA EXPORT
# =============================================================================

class DataExporter:
    """Export data to CSV files"""
    
    @staticmethod
    def export_to_csv(data: List[Dict], filename: str):
        """Export list of dictionaries to CSV"""
        if not data:
            logger.warning(f"No data to export to {filename}")
            return
        
        df = pd.DataFrame(data)
        
        # Reorder columns for better readability
        priority_columns = ['title', 'price', 'original_price_clean', 'mmr_price', 
                          'price_difference', 'price_difference_pct', 'vin', 'mileage', 
                          'phone_numbers', 'phone_count', 'is_qualified_lead', 
                          'listing_url', 'scraped_at']
        
        existing_priority = [col for col in priority_columns if col in df.columns]
        other_columns = [col for col in df.columns if col not in priority_columns]
        column_order = existing_priority + other_columns
        
        df = df[column_order]
        df.to_csv(filename, index=False, encoding='utf-8')
        logger.info(f"✓ Exported {len(data)} records to {filename}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class LeadGenerationPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self):
        self.config = PipelineConfig()
        self.validate_config()
    
    def validate_config(self):
        """Validate required environment variables"""
        missing = []
        if not self.config.TWILIO_ACCOUNT_SID:
            missing.append('TWILIO_ACCOUNT_SID')
        if not self.config.TWILIO_AUTH_TOKEN:
            missing.append('TWILIO_AUTH_TOKEN')
        if not self.config.TWILIO_PHONE_NUMBER:
            missing.append('TWILIO_PHONE_NUMBER')
        if not self.config.OPENAI_API_KEY:
            missing.append('OPENAI_API_KEY')
        if not self.config.MANHEIM_ENDPOINT:
            missing.append('MANHEIM_ENDPOINT')
        if not self.config.MANHEIM_AUTH:
            missing.append('MANHEIM_AUTH')
        if not self.config.MANHEIM_API_BASE_URL:
            missing.append('MANHEIM_API_BASE_URL')
        
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    
    def run(self, skip_scraping=False, skip_salesforce=False, skip_sms=False):
        """
        Run the complete pipeline
        
        Args:
            skip_scraping: Skip web scraping (use existing CSV)
            skip_salesforce: Skip Salesforce submission
            skip_sms: Skip SMS outreach
        """
        logger.info("=" * 70)
        logger.info("LEAD GENERATION PIPELINE STARTING")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        # Step 1: Scrape or load data
        if not skip_scraping:
            scraper = CraigslistScraper(self.config)
            vehicles = scraper.scrape_listings()
        else:
            logger.info("Skipping scraping - loading from CSV")
            # Implement CSV loading logic here if needed
            vehicles = []
        
        if not vehicles:
            logger.error("No vehicles to process. Exiting.")
            return
        
        # Step 2: Extract phone numbers
        vehicles = PhoneExtractor.add_phones_to_vehicles(vehicles)
        
        # Step 3: Get MMR pricing
        mmr_service = MMRPricingService(self.config)
        vehicles = mmr_service.add_mmr_pricing(vehicles)
        
        # Step 4: Qualify leads
        qualified_leads, non_leads = LeadQualifier.qualify_leads(vehicles)
        
        # Show phone number statistics
        vehicles_with_phones = sum(1 for v in vehicles if v.get('phone_count', 0) > 0)
        logger.info(f"Vehicles with phone numbers: {vehicles_with_phones}/{len(vehicles)}")
        
        if vehicles_with_phones == 0:
            logger.warning("=" * 70)
            logger.warning("WARNING: No phone numbers were extracted!")
            logger.warning("This could mean:")
            logger.warning("  1. The 'show-contact' button click isn't working")
            logger.warning("  2. Phone numbers are not posted in these listings")
            logger.warning("  3. Craigslist changed their page structure")
            logger.warning("Check the scraped_vehicles CSV to verify the 'vehicle_details' field")
            logger.warning("=" * 70)
        
        # Export all data
        DataExporter.export_to_csv(vehicles, self.config.SCRAPED_DATA_FILE)
        DataExporter.export_to_csv(qualified_leads, self.config.QUALIFIED_LEADS_FILE)
        
        # Step 5: Send to Salesforce
        if not skip_salesforce and qualified_leads:
            salesforce = SalesforceIntegration(self.config)
            salesforce.send_all_leads(qualified_leads)
        
        # Step 6: SMS outreach
        if not skip_sms and qualified_leads:
            sms = SMSOutreach(self.config)
            sms.contact_qualified_leads(qualified_leads)
        
        # Final summary
        elapsed = time.time() - start_time
        
        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETED")
        logger.info("=" * 70)
        logger.info(f"Total Execution Time: {elapsed/60:.2f} minutes")
        logger.info(f"Total Vehicles Scraped: {len(vehicles)}")
        logger.info(f"Vehicles with Phone Numbers: {sum(1 for v in vehicles if v.get('phone_count', 0) > 0)}")
        logger.info(f"Qualified Leads: {len(qualified_leads)}")
        logger.info(f"Non-Leads: {len(non_leads)}")
        logger.info(f"Data exported to:")
        logger.info(f"  - {self.config.SCRAPED_DATA_FILE}")
        logger.info(f"  - {self.config.QUALIFIED_LEADS_FILE}")
        logger.info("=" * 70)


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Main entry point with command line options"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Integrated Vehicle Lead Generation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python integrated_lead_pipeline.py
  
  # Skip scraping (use existing data)
  python integrated_lead_pipeline.py --skip-scraping
  
  # Skip Salesforce submission
  python integrated_lead_pipeline.py --skip-salesforce
  
  # Skip SMS outreach
  python integrated_lead_pipeline.py --skip-sms
  
  # Run scraping and qualification only
  python integrated_lead_pipeline.py --skip-salesforce --skip-sms
  
  # Run in headless mode
  python integrated_lead_pipeline.py --headless
        """
    )
    
    parser.add_argument(
        '--skip-scraping',
        action='store_true',
        help='Skip web scraping and use existing CSV data'
    )
    
    parser.add_argument(
        '--skip-salesforce',
        action='store_true',
        help='Skip sending leads to Salesforce'
    )
    
    parser.add_argument(
        '--skip-sms',
        action='store_true',
        help='Skip SMS outreach to qualified leads'
    )
    
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run browser in headless mode'
    )
    
    parser.add_argument(
        '--url',
        type=str,
        help='Override target Craigslist URL'
    )
    
    args = parser.parse_args()
    
    # Configure based on arguments
    if args.headless:
        PipelineConfig.HEADLESS_MODE = True
    
    if args.url:
        PipelineConfig.TARGET_URL = args.url
    
    # Run pipeline
    try:
        pipeline = LeadGenerationPipeline()
        pipeline.run(
            skip_scraping=args.skip_scraping,
            skip_salesforce=args.skip_salesforce,
            skip_sms=args.skip_sms
        )
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())