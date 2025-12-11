"""
Integrated Vehicle Lead Generation Pipeline
Combines all functionalities: scraping, phone extraction, MMR pricing, lead qualification, and outreach
"""

import os
import time
import stat
import csv
import re
import json
import logging
import subprocess
import pandas as pd
import requests
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable
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
    from database import save_message, phone_number_exists
except ImportError:
    # If database module is not available, create a no-op function
    def save_message(*args, **kwargs):
        logger.warning("Database module not available - SMS messages will not be saved to database")
        return False
    def phone_number_exists(*args, **kwargs):
        logger.warning("Database module not available - cannot check if phone number exists")
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
        "Hi, I'm interested in your car for sale at your asking price. Do you mind sending me the vin so I can run a carfax on it?")
    
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
    
    def __init__(self, config: PipelineConfig, stop_check: Optional[Callable[[], bool]] = None):
        """
        Initialize the Craigslist scraper.
        
        Args:
            config: Pipeline configuration object
            stop_check: Optional callback function that returns True if scraping should stop
        """
        self.config = config
        self.driver = None
        self.wait = None
        self._clicked_show_contact_recently = False
        self.stop_check = stop_check
        self._setup_chrome_driver()
    
    def _setup_chrome_driver(self):
        """Setup Chrome WebDriver"""
        logger.info("Setting up Chrome WebDriver...")
        
        chrome_options = Options()
        if self.config.HEADLESS_MODE:
            chrome_options.add_argument("--headless=new")
        
        # Essential Docker/container options
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-software-rasterizer")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-background-timer-throttling")
        chrome_options.add_argument("--disable-backgrounding-occluded-windows")
        chrome_options.add_argument("--disable-renderer-backgrounding")
        chrome_options.add_argument("--disable-features=TranslateUI")
        chrome_options.add_argument("--disable-ipc-flooding-protection")
        
        # Additional stability options
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        # Set preferences to avoid crashes
        prefs = {
            "profile.default_content_setting_values.notifications": 2,
            "profile.default_content_settings.popups": 0,
            "profile.managed_default_content_settings.images": 1,
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        # Try to use system chromium/chromedriver first (for Docker/Railway)
        chromium_binary = None
        for binary_path in ["/usr/bin/chromium", "/usr/bin/chromium-browser", "/usr/bin/google-chrome"]:
            if os.path.exists(binary_path):
                chromium_binary = binary_path
                chrome_options.binary_location = chromium_binary
                logger.info(f"Using browser binary: {chromium_binary}")
                break
        
        # Try system chromedriver first (common locations in Docker/Ubuntu)
        system_chromedriver = None
        import shutil
        
        # First, try to find in PATH
        chromedriver_in_path = shutil.which("chromedriver")
        if chromedriver_in_path and os.path.exists(chromedriver_in_path):
            system_chromedriver = chromedriver_in_path
            logger.info(f"Found chromedriver in PATH: {system_chromedriver}")
        
        # If not in PATH, check common locations
        if not system_chromedriver:
            for driver_path in [
                "/usr/bin/chromedriver",
                "/usr/lib/chromium/chromedriver",
                "/usr/lib/chromium-browser/chromedriver",
                "/usr/lib64/chromium/chromedriver",
                "/usr/local/bin/chromedriver"
            ]:
                if os.path.exists(driver_path):
                    # Try to make it executable if it's not
                    if not os.access(driver_path, os.X_OK):
                        try:
                            os.chmod(driver_path, 0o755)
                            logger.info(f"Made chromedriver executable: {driver_path}")
                        except Exception as e:
                            logger.warning(f"Could not make {driver_path} executable: {e}")
                    
                    if os.access(driver_path, os.X_OK):
                        system_chromedriver = driver_path
                        logger.info(f"Using system chromedriver: {system_chromedriver}")
                        break
        
        using_system_driver = False
        if system_chromedriver:
            driver_path = system_chromedriver
            using_system_driver = True
        else:
            # Fallback to ChromeDriverManager (but this will likely fail in Docker)
            logger.warning("System chromedriver not found, attempting ChromeDriverManager (may fail in Docker)")
            try:
                driver_path = ChromeDriverManager().install()
                logger.info(f"ChromeDriverManager installed driver at: {driver_path}")
            except Exception as e:
                logger.error(f"ChromeDriverManager failed: {e}")
                # Final attempt: search specific known locations for chromedriver
                logger.info("Performing final search for chromedriver in known locations...")
                found_driver = None
                # Search in specific subdirectories to avoid slow full tree walks
                search_locations = [
                    "/usr/lib/chromium",
                    "/usr/lib/chromium-browser",
                    "/usr/lib64/chromium",
                    "/usr/local/bin",
                    "/usr/local/lib",
                    "/opt/chromium",
                ]
                for search_path in search_locations:
                    if os.path.exists(search_path):
                        for root, dirs, files in os.walk(search_path):
                            # Limit depth to avoid slow searches
                            depth = root[len(search_path):].count(os.sep)
                            if depth > 2:  # Max 2 levels deep
                                dirs[:] = []  # Don't recurse deeper
                                continue
                            
                            if "chromedriver" in files:
                                candidate = os.path.join(root, "chromedriver")
                                if os.path.isfile(candidate):
                                    try:
                                        os.chmod(candidate, 0o755)
                                        found_driver = candidate
                                        logger.info(f"Found chromedriver via filesystem search: {found_driver}")
                                        break
                                    except Exception:
                                        continue
                            if found_driver:
                                break
                    if found_driver:
                        break
                
                if found_driver:
                    driver_path = found_driver
                    using_system_driver = True
                else:
                    raise RuntimeError(
                        "Could not find chromedriver. Please ensure chromium-driver is installed. "
                        "Checked: PATH, /usr/bin, /usr/lib/chromium, and performed filesystem search."
                    )

        # Only process ChromeDriverManager paths (skip for system drivers)
        if not using_system_driver:
            if os.path.basename(driver_path).startswith("THIRD_PARTY_NOTICES"):
                driver_path = os.path.dirname(driver_path)

            if os.path.isdir(driver_path):
                candidate = os.path.join(driver_path, "chromedriver")
                if os.path.isfile(candidate):
                    driver_path = candidate
                else:
                    resolved = None
                    for root, _dirs, files in os.walk(driver_path):
                        for filename in files:
                            if filename in {"chromedriver", "chromedriver.exe"}:
                                full_path = os.path.join(root, filename)
                                if os.path.isfile(full_path):
                                    resolved = full_path
                                    break
                        if resolved:
                            break
                    if resolved:
                        driver_path = resolved

        if os.path.isfile(driver_path) and not os.access(driver_path, os.X_OK):
            try:
                current_mode = os.stat(driver_path).st_mode
                os.chmod(driver_path, current_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
            except OSError as chmod_error:
                raise PermissionError(f"Found chromedriver at {driver_path} but could not mark it executable: {chmod_error}")

        if not os.path.isfile(driver_path) or not os.access(driver_path, os.X_OK):
            raise FileNotFoundError(f"Unable to locate executable chromedriver in {driver_path}")

        driver_dir = os.path.dirname(driver_path)
        os.environ["PATH"] = f"{driver_dir}{os.pathsep}" + os.environ.get("PATH", "")

        # Create Service with explicit log output handling to avoid closed file handle issues
        # Use subprocess.DEVNULL to prevent file handle conflicts in Docker environments
        service = Service(
            executable_path=driver_path,
            log_output=subprocess.DEVNULL  # Prevent file handle issues
        )
        
        # Try to create driver, with fallback to single-process mode if it fails
        try:
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            logger.info("Chrome WebDriver initialized successfully")
        except Exception as e:
            logger.warning(f"Initial Chrome driver creation failed: {e}")
            logger.info("Retrying with --single-process mode (Docker compatibility)...")
            # Create a fresh Service object for retry to avoid closed file handle issues
            service_retry = Service(
                executable_path=driver_path,
                log_output=subprocess.DEVNULL
            )
            # Add single-process option as fallback
            chrome_options.add_argument("--single-process")
            try:
                self.driver = webdriver.Chrome(service=service_retry, options=chrome_options)
                logger.info("Chrome WebDriver initialized successfully with --single-process")
            except Exception as e2:
                logger.error(f"Chrome driver creation failed even with --single-process: {e2}")
                raise RuntimeError(
                    f"Failed to initialize Chrome WebDriver. "
                    f"First attempt: {e}. Second attempt (with --single-process): {e2}. "
                    f"Ensure Chromium and chromedriver are properly installed and compatible."
                ) from e2
        
        self.wait = WebDriverWait(self.driver, 15)
    
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
                # Check for stop request before processing each listing
                if self.stop_check and self.stop_check():
                    logger.info(f"Stop requested. Processed {len(vehicles)}/{total_cards} listings before stopping.")
                    break
                
                try:
                    logger.info(f"Processing listing {index + 1}/{total_cards}")
                    
                    # Re-fetch cards to avoid stale references
                    current_cards = self._get_gallery_cards()
                    if index >= len(current_cards):
                        break
                    
                    card = current_cards[index]
                    
                    # Extract location from card BEFORE clicking (using xpath: //div[@class='meta']/text()[last()])
                    location = self._extract_location_from_card(card)
                    logger.info(f"Extracted location: {location}")
                    
                    # Get listing URL before clicking
                    listing_url = self._get_listing_url(card)
                    
                    # Click and scrape
                    if self._click_gallery_card(card):
                        vehicle_data = self._scrape_vehicle_details()
                        vehicle_data['listing_url'] = listing_url
                        vehicle_data['location'] = location  # Add location to vehicle data
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
    
    def _extract_location_from_card(self, card_element) -> str:
        """
        Extract location from gallery card before clicking.
        Uses xpath: //div[@class='meta']/text()[last()]
        
        Args:
            card_element: The gallery card WebElement
            
        Returns:
            str: Location text or "Not Found" if extraction fails
        """
        try:
            # Find the meta div within the card element
            meta_div = card_element.find_element(By.XPATH, ".//div[@class='meta']")
            
            # Get all text from the meta div
            meta_text = meta_div.text.strip()
            
            # The xpath text()[last()] suggests we want the last text node
            # In Selenium, we can get all text and split, or use JavaScript to get last text node
            # For simplicity, we'll try to get the last part of the text
            # If there are multiple text segments, take the last one
            if meta_text:
                # Split by common separators and take the last meaningful segment
                text_parts = [part.strip() for part in meta_text.split('\n') if part.strip()]
                if text_parts:
                    location = text_parts[-1]  # Get last text segment
                    logger.debug(f"Extracted location from card: {location}")
                    return location
            
            # Alternative: Use JavaScript to get the last text node directly
            try:
                last_text_node = self.driver.execute_script(
                    """
                    var meta = arguments[0].querySelector('.meta');
                    if (!meta) return '';
                    var textNodes = [];
                    var walker = document.createTreeWalker(
                        meta,
                        NodeFilter.SHOW_TEXT,
                        null,
                        false
                    );
                    var node;
                    while (node = walker.nextNode()) {
                        if (node.textContent.trim()) {
                            textNodes.push(node.textContent.trim());
                        }
                    }
                    return textNodes.length > 0 ? textNodes[textNodes.length - 1] : '';
                    """,
                    card_element
                )
                if last_text_node:
                    logger.debug(f"Extracted location via JavaScript: {last_text_node}")
                    return last_text_node
            except Exception as js_error:
                logger.debug(f"JavaScript extraction failed: {js_error}")
            
            return "Not Found"
        except NoSuchElementException:
            logger.debug("Meta div not found in card")
            return "Not Found"
        except Exception as e:
            logger.warning(f"Error extracting location from card: {str(e)}")
            return "Not Found"
    
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
                if text:
                    extracted = PhoneExtractor.extract_phones(text)
                    if extracted:
                        logger.info(f"Found potential phone: {extracted}")
                        return extracted
                
                # Check href attribute for tel: links
                href = element.get_attribute('href')
                if href and 'tel:' in href:
                    phone = PhoneExtractor.extract_phones(href)
                    if phone:
                        logger.info(f"Found phone in tel link: {phone}")
                        return phone
            except:
                continue
        
        # Last resort: check entire page source for phone patterns
        try:
            page_source = self.driver.page_source
            extracted = PhoneExtractor.extract_phones(page_source)
            if extracted:
                logger.info(f"Found phone in page source: {extracted}")
                return extracted
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
        """
        Close browser and cleanup resources.
        
        Ensures proper cleanup of Chrome processes to prevent resource leaks
        and file handle issues in subsequent requests.
        """
        if self.driver:
            try:
                self.driver.quit()
                logger.info("Browser closed successfully")
            except Exception as e:
                logger.warning(f"Error closing browser: {e}")
            finally:
                # Ensure driver reference is cleared
                self.driver = None
                self.wait = None


# =============================================================================
# STEP 2: PHONE NUMBER EXTRACTION
# =============================================================================

class PhoneExtractor:
    """Extract phone numbers from vehicle details"""
    
    @staticmethod
    def extract_phones(text: str) -> str:
        """
        Extract phone numbers from text
        Returns single phone number string (not a list)
        """
        if not text:
            return ""
        
        text = text.replace('\u2013', '-').replace('\u2014', '-')
        candidate_data = {}
        pattern = re.compile(r'\(?([2-9]\d{2})\)?[-.\s]?([2-9]\d{2})[-.\s]?(\d{4})')
        positive_context = ['contact', 'call', 'text', 'mensaje', 'message', 'phone', 'reach', 'llamar', 'se habla', 'cel', 'whatsapp']
        negative_context = ['post id', 'posting id', 'postid', 'stock #', 'stock#']
        
        for match in pattern.finditer(text):
            digits = ''.join(match.groups())
            if len(digits) != 10:
                continue
            if digits.startswith('789'):
                continue
            
            formatted = f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
            start, end = match.start(), match.end()
            raw_match = match.group(0)
            context_before = text[max(0, start - 40):start].lower()
            context_after = text[start:end + 40].lower()
            
            score = 0
            if '(' in raw_match or ')' in raw_match:
                score += 2
            if any(keyword in context_before for keyword in positive_context):
                score += 3
            if any(keyword in context_after for keyword in positive_context):
                score += 2
            if any(keyword in context_before for keyword in negative_context) or any(keyword in context_after for keyword in negative_context):
                score -= 6
            
            existing = candidate_data.get(digits)
            candidate = {
                'formatted': formatted,
                'score': score,
                'position': start
            }
            
            if existing is None or candidate['score'] > existing['score'] or (
                candidate['score'] == existing['score'] and candidate['position'] < existing['position']
            ):
                candidate_data[digits] = candidate
        
        if not candidate_data:
            return ""
        
        best_candidate = max(
            candidate_data.values(),
            key=lambda item: (item['score'], -item['position'])
        )
        
        if best_candidate['score'] <= 0:
            return ""
        
        return best_candidate['formatted']
    
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
            phone = PhoneExtractor.extract_phones(details)
            vehicle['phone_numbers'] = phone
            
            if phone:
                logger.info(f"Found phone for: {vehicle['title'][:50]} - {phone}")
        
        total_with_phones = sum(1 for v in vehicles if v.get('phone_numbers'))
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
        Separate qualified leads from non-leads.
        
        Qualification Criteria:
        1. For vehicles WITH VIN:
           - Must have price and MMR
           - Must have phone number (disqualify if VIN exists but no phone)
           - Price <= Final MMR Price (where Final MMR = Base MMR * 1.10)
           - These leads go to SALESFORCE ONLY (no SMS)
        
        2. For vehicles WITHOUT VIN but WITH PHONE:
           - Must have phone number
           - These leads are qualified for SMS outreach (no MMR check needed)
        
        3. For vehicles WITHOUT VIN and WITHOUT PHONE:
           - Disqualified (cannot contact)
        
        IMPORTANT: 
        - Vehicles with VIN + Phone + qualified = Salesforce only (skip SMS)
        - Vehicles with Phone but no VIN = SMS outreach (no Salesforce)
        - Vehicles with VIN but no phone = Disqualified
        """
        logger.info("=" * 70)
        logger.info("QUALIFYING LEADS")
        logger.info("=" * 70)
        
        qualified_leads = []
        non_leads = []
        
        for vehicle in vehicles:
            original_price = LeadQualifier.clean_price(vehicle.get('price', ''))
            mmr_price = vehicle.get('mmr_price')
            vin = vehicle.get('vin', '').strip()
            phone_number = vehicle.get('phone_numbers', '').strip() if vehicle.get('phone_numbers') else ''
            
            has_vin = vin and vin.lower() != 'not found'
            has_phone = bool(phone_number)
            
            # CASE 1: Vehicle has VIN
            if has_vin:
                # If VIN exists but no phone, disqualify
                if not has_phone:
                    vehicle['is_qualified_lead'] = False
                    vehicle['disqualification_reason'] = 'Has VIN but no phone number'
                    non_leads.append(vehicle)
                    logger.info(f"[DISQUALIFIED] {vehicle.get('title', 'Unknown')[:50]}")
                    logger.info(f"  Reason: Has VIN ({vin[:17]}...) but no phone number - cannot contact seller")
                    continue
                
                # Vehicle has both VIN and phone - check price vs MMR
                if original_price is None or mmr_price is None:
                    logger.debug(f"Disqualified - Has VIN but missing price or MMR: {vehicle.get('title', 'Unknown')[:50]}")
                    non_leads.append(vehicle)
                    continue
                
                vehicle['original_price_clean'] = original_price
                
                # Calculate final MMR price (MMR * 1.10)
                mmr_price_final = mmr_price * 1.10
                vehicle['mmr_price_adjusted'] = mmr_price_final
                vehicle['mmr_price_final'] = mmr_price_final
                
                # Qualify if price <= final MMR
                if original_price <= mmr_price_final:
                    vehicle['is_qualified_lead'] = True
                    vehicle['salesforce_only'] = True  # Mark for Salesforce only (no SMS)
                    qualified_leads.append(vehicle)
                    logger.info(f"[QUALIFIED LEAD - SALESFORCE ONLY] {vehicle.get('title', 'Unknown')[:50]}")
                    logger.info(f"  Listing Price: ${original_price:,.0f} | Base MMR: ${mmr_price:,.0f} | Final MMR (x1.10): ${mmr_price_final:,.0f}")
                    logger.info(f"  VIN: {vin[:17]}... | Phone: {phone_number} | Action: Salesforce only (SMS skipped)")
                else:
                    vehicle['is_qualified_lead'] = False
                    vehicle['disqualification_reason'] = 'Listing price exceeds final MMR price'
                    non_leads.append(vehicle)
            
            # CASE 2: Vehicle has phone but NO VIN - qualify for SMS outreach
            elif has_phone:
                vehicle['is_qualified_lead'] = True
                vehicle['sms_only'] = True  # Mark for SMS only (no Salesforce, no MMR check)
                qualified_leads.append(vehicle)
                logger.info(f"[QUALIFIED LEAD - SMS ONLY] {vehicle.get('title', 'Unknown')[:50]}")
                logger.info(f"  Phone: {phone_number} | No VIN - qualified for SMS outreach")
                if original_price:
                    logger.info(f"  Listing Price: ${original_price:,.0f} (MMR not available - no VIN)")
            
            # CASE 3: Vehicle has neither VIN nor phone - disqualify
            else:
                vehicle['is_qualified_lead'] = False
                vehicle['disqualification_reason'] = 'No VIN and no phone number - cannot contact'
                non_leads.append(vehicle)
                logger.debug(f"Disqualified - No VIN and no phone: {vehicle.get('title', 'Unknown')[:50]}")
        
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
        """
        Send a single lead to Salesforce.
        
        Requirements:
        - Lead must have a phone number (required for Salesforce submission)
        - Lead must be qualified (price <= MMR_price * 1.10)
        
        Args:
            vehicle: Vehicle dictionary with lead information
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        # Validate phone number requirement - CRITICAL: No phone = No Salesforce submission
        primary_phone = vehicle.get('phone_numbers', '')
        if not primary_phone or not primary_phone.strip():
            logger.warning(f"Cannot send to Salesforce (no phone number): {vehicle.get('title', 'Unknown')[:50]}")
            return False
        
        # Parse name from title (fallback to generic if not found)
        first_name, last_name = self._parse_name(vehicle['title'])
        
        payload = {
            "contact": {
                "first_name": first_name,
                "last_name": last_name,
                "email": "",  # Not available from scraping
                "phone": primary_phone
            },
            "vehicle": {
                "vin": vehicle.get('vin', ''),
                "mileage": vehicle.get('mileage', '').replace(',', ''),
                "car_location": vehicle.get('location', '')  # Location extracted from gallery card
            },
             "lead":{
                "source": "CLB"
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
                location_info = f" | Location: {vehicle.get('location', 'N/A')}" if vehicle.get('location') else ""
                logger.info(f"✓ Successfully sent to Salesforce: {vehicle['title'][:50]}{location_info}")
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
        """
        Send qualified leads to Salesforce.
        
        IMPORTANT RULES:
        1. Only leads WITH VIN will be sent to Salesforce (leads without VIN are SMS-only)
        2. Leads must have phone number (required)
        3. Leads must be qualified (price <= MMR_price * 1.10 if VIN exists)
        
        Leads without VIN are skipped from Salesforce (they are SMS-only leads).
        Leads without phone numbers are skipped even if they have VIN.
        
        Args:
            qualified_leads: List of qualified lead dictionaries
            
        Returns:
            Dict with 'sent', 'failed', and 'skipped' counts
        """
        logger.info("=" * 70)
        logger.info("SENDING LEADS TO SALESFORCE")
        logger.info("=" * 70)
        logger.info("NOTE: Only leads with VIN will be sent to Salesforce")
        logger.info("NOTE: Leads without VIN are SMS-only (not sent to Salesforce)")
        logger.info("=" * 70)
        
        results = {
            'sent': 0,
            'failed': 0,
            'skipped': 0,
            'no_vin_skipped': 0  # Track leads without VIN (SMS-only)
        }
        
        # Count leads without VIN before processing
        leads_without_vin = sum(1 for lead in qualified_leads 
                               if not lead.get('vin') or lead.get('vin', '').strip().lower() == 'not found')
        
        if leads_without_vin > 0:
            logger.info(f"Found {leads_without_vin} qualified lead(s) without VIN - these are SMS-only (skipped from Salesforce)")
        
        # Count leads without phone numbers before processing
        leads_without_phone = sum(1 for lead in qualified_leads 
                                  if not lead.get('phone_numbers') or not str(lead.get('phone_numbers', '')).strip())
        
        if leads_without_phone > 0:
            logger.info(f"Found {leads_without_phone} qualified lead(s) without phone numbers - these will be skipped")
        
        for lead in qualified_leads:
            vin = lead.get('vin', '').strip()
            has_vin = vin and vin.lower() != 'not found'
            
            # SKIP leads without VIN - these are SMS-only, not Salesforce leads
            if not has_vin:
                logger.debug(f"Skipping Salesforce (no VIN - SMS only): {lead.get('title', 'Unknown')[:50]}")
                results['no_vin_skipped'] += 1
                continue
            
            # Validate phone number requirement - CRITICAL CHECK
            phone_number = lead.get('phone_numbers', '')
            if not phone_number or not str(phone_number).strip():
                logger.warning(f"Skipping Salesforce submission (no phone number): {lead.get('title', 'Unknown')[:50]} | VIN: {vin[:17]}...")
                results['skipped'] += 1
                continue
            
            # Send lead to Salesforce (lead has both VIN and phone)
            logger.info(f"Sending to Salesforce: {lead.get('title', 'Unknown')[:50]} | VIN: {vin[:17]}... | Phone: {phone_number}")
            if self.send_lead(lead):
                results['sent'] += 1
            else:
                results['failed'] += 1
            
            time.sleep(self.config.API_DELAY)
        
        logger.info(f"Salesforce Results: {results['sent']} sent, {results['failed']} failed, {results['skipped']} skipped (no phone), {results['no_vin_skipped']} skipped (no VIN - SMS only)")
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
                logger.info(f"✓ Saved SMS message to database for {formatted_number}")
            except Exception as db_error:
                # CRITICAL: Log error prominently so user knows messages aren't being saved
                # SMS was sent successfully via Twilio, but won't appear in frontend without database
                logger.error(f"✗ CRITICAL: Failed to save SMS to database for {formatted_number}")
                logger.error(f"  Error: {str(db_error)}")
                logger.error(f"  SMS was sent successfully via Twilio, but will NOT appear in frontend!")
                logger.error(f"  Please fix database connection to save future messages.")
            
            return True
        except Exception as e:
            logger.error(f"✗ Failed to send SMS to {to_number}: {str(e)}")
            return False
    
    def contact_qualified_leads(self, qualified_leads: List[Dict]) -> Dict:
        """
        Send SMS to qualified leads.
        
        IMPORTANT RULES:
        1. SKIP leads with VIN (these go to Salesforce only, not SMS)
        2. Only send SMS to leads with phone but NO VIN
        3. Before sending, check if phone number already exists in database
        4. Skip SMS if phone number was already contacted
        
        Only sends SMS to:
        - Leads with phone number
        - Leads WITHOUT VIN (leads with VIN are Salesforce-only)
        - Phone numbers NOT already in database
        """
        logger.info("=" * 70)
        logger.info("SENDING SMS TO QUALIFIED LEADS")
        logger.info("=" * 70)
        logger.info("NOTE: Leads with VIN are skipped (Salesforce only)")
        logger.info("NOTE: Phone numbers already in database will be skipped")
        logger.info("=" * 70)
        
        results = {
            'sent': 0,
            'failed': 0,
            'no_phone': 0,
            'has_vin_skipped': 0,  # Track leads with VIN that are skipped (Salesforce only)
            'already_exists': 0  # Track phone numbers that already exist in DB
        }
        
        for lead in qualified_leads:
            phone = lead.get('phone_numbers', '')
            vin = lead.get('vin', '').strip()
            has_vin = vin and vin.lower() != 'not found'
            
            # SKIP leads with VIN - these go to Salesforce only, not SMS
            if has_vin:
                logger.info(f"⏭ Skipping SMS - Lead has VIN (Salesforce only): {lead.get('title', 'Unknown')[:50]}")
                logger.info(f"   VIN: {vin[:17]}... | Phone: {phone}")
                results['has_vin_skipped'] += 1
                continue
            
            # Only process leads without VIN
            if not phone:
                results['no_phone'] += 1
                continue
            
            # Format phone number to E.164 for consistent database checking
            formatted_number = self.format_phone(phone)
            
            # Check if phone number already exists in database
            try:
                if phone_number_exists(formatted_number):
                    logger.info(f"⏭ Skipping {formatted_number} - phone number already exists in database")
                    logger.info(f"   Vehicle: {lead['title'][:50]}")
                    results['already_exists'] += 1
                    continue
            except Exception as check_error:
                # If check fails, log warning but continue (fail-safe approach)
                logger.warning(f"Error checking if phone number exists: {check_error}. Proceeding with SMS send.")
            
            logger.info(f"Contacting: {lead['title'][:50]} | Phone: {formatted_number} (No VIN - SMS outreach)")
            
            if self.send_sms(phone, self.config.INITIAL_MESSAGE):
                results['sent'] += 1
            else:
                results['failed'] += 1
            
            time.sleep(self.config.API_DELAY)
        
        logger.info(f"SMS Results: {results['sent']} sent, {results['failed']} failed, {results['no_phone']} no phone, {results['has_vin_skipped']} skipped (has VIN - Salesforce only), {results['already_exists']} already in DB (skipped)")
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
        
        # Remove unwanted columns
        columns_to_remove = ['price_difference', 'price_difference_pct', 'phone_count', 'scraped_at']
        for col in columns_to_remove:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # Reorder columns for better readability
        priority_columns = ['title', 'price', 'original_price_clean', 'mmr_price', 'mmr_price_final', 'mmr_price_adjusted',
                          'vin', 'mileage', 'location', 'phone_numbers', 'is_qualified_lead', 
                          'listing_url']
        
        existing_priority = [col for col in priority_columns if col in df.columns]
        other_columns = [col for col in df.columns if col not in priority_columns]
        column_order = existing_priority + other_columns
        
        df = df[column_order]
        # Use default pandas CSV export with QUOTE_MINIMAL (automatically quotes fields with special characters)
        # This properly handles commas, newlines, and quotes in fields like vehicle_details
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
    
    def run(self, skip_scraping: bool = False, skip_salesforce: bool = False, skip_sms: bool = False, stop_check: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
        """
        Run the complete pipeline
        
        Args:
            skip_scraping: Skip web scraping (use existing CSV)
            skip_salesforce: Skip Salesforce submission
            skip_sms: Skip SMS outreach
        Returns:
            Dict containing execution summary details including counts and export paths.
        """
        logger.info("=" * 70)
        logger.info("LEAD GENERATION PIPELINE STARTING")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        # Step 1: Scrape or load data
        scraper = None
        try:
            if not skip_scraping:
                # Check for stop request before starting scraping
                if stop_check and stop_check():
                    logger.info("Stop requested before scraping started")
                    return {
                        'status': 'stopped',
                        'vehicles_scraped': 0,
                        'vehicles_with_phone_numbers': 0,
                        'qualified_leads': 0,
                        'non_leads': 0,
                        'scraped_data_file': None,
                        'qualified_leads_file': None,
                        'salesforce_results': None,
                        'sms_results': None,
                        'elapsed_seconds': time.time() - start_time,
                    }
                
                scraper = CraigslistScraper(self.config, stop_check=stop_check)
                vehicles = scraper.scrape_listings()
            else:
                logger.info("Skipping scraping - loading from CSV")
                # Implement CSV loading logic here if needed
                vehicles = []
        finally:
            # Ensure scraper is always closed to prevent resource leaks
            if scraper:
                try:
                    scraper.close()
                except Exception as cleanup_error:
                    logger.warning(f"Error during scraper cleanup: {cleanup_error}")
        
        # Check for stop request after scraping
        if stop_check and stop_check():
            logger.info("Stop requested after scraping")
            return {
                'status': 'stopped',
                'vehicles_scraped': len(vehicles) if vehicles else 0,
                'vehicles_with_phone_numbers': 0,
                'qualified_leads': 0,
                'non_leads': 0,
                'scraped_data_file': None,
                'qualified_leads_file': None,
                'salesforce_results': None,
                'sms_results': None,
                'elapsed_seconds': time.time() - start_time,
            }
        
        if not vehicles:
            logger.error("No vehicles to process. Exiting.")
            return {
                'status': 'no_data',
                'vehicles_scraped': 0,
                'vehicles_with_phone_numbers': 0,
                'qualified_leads': 0,
                'non_leads': 0,
                'scraped_data_file': None,
                'qualified_leads_file': None,
                'salesforce_results': None,
                'sms_results': None,
                'elapsed_seconds': time.time() - start_time,
            }
        
        # Step 2: Extract phone numbers
        vehicles = PhoneExtractor.add_phones_to_vehicles(vehicles)
        
        # Step 3: Get MMR pricing
        mmr_service = MMRPricingService(self.config)
        vehicles = mmr_service.add_mmr_pricing(vehicles)
        
        # Step 4: Qualify leads
        qualified_leads, non_leads = LeadQualifier.qualify_leads(vehicles)
        
        # Show phone number statistics
        # Note: phone_numbers is a string (first phone found), not a count
        vehicles_with_phones = sum(1 for v in vehicles if v.get('phone_numbers', '').strip())
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
        
        summary: Dict[str, Any] = {
            'status': 'success',
            'vehicles_scraped': len(vehicles),
            'vehicles_with_phone_numbers': vehicles_with_phones,
            'qualified_leads': len(qualified_leads),
            'non_leads': len(non_leads),
            'scraped_data_file': self.config.SCRAPED_DATA_FILE,
            'qualified_leads_file': self.config.QUALIFIED_LEADS_FILE,
            'salesforce_results': None,
            'sms_results': None,
            'elapsed_seconds': None,
        }

        # Step 5: Send to Salesforce
        if not skip_salesforce and qualified_leads:
            salesforce = SalesforceIntegration(self.config)
            summary['salesforce_results'] = salesforce.send_all_leads(qualified_leads)
        
        # Step 6: SMS outreach
        if not skip_sms and qualified_leads:
            sms = SMSOutreach(self.config)
            summary['sms_results'] = sms.contact_qualified_leads(qualified_leads)
        
        # Final summary
        elapsed = time.time() - start_time
        summary['elapsed_seconds'] = elapsed
        
        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETED")
        logger.info("=" * 70)
        logger.info(f"Total Execution Time: {elapsed/60:.2f} minutes")
        logger.info(f"Total Vehicles Scraped: {len(vehicles)}")
        logger.info(f"Vehicles with Phone Numbers: {vehicles_with_phones}")
        logger.info(f"Qualified Leads: {len(qualified_leads)}")
        logger.info(f"Non-Leads: {len(non_leads)}")
        logger.info(f"Data exported to:")
        logger.info(f"  - {self.config.SCRAPED_DATA_FILE}")
        logger.info(f"  - {self.config.QUALIFIED_LEADS_FILE}")
        logger.info("=" * 70)

        return summary


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
  python all_rounder_2.py
  
  # Skip scraping (use existing data)
  python all_rounder_2.py --skip-scraping
  
  # Skip Salesforce submission
  python all_rounder_2.py --skip-salesforce
  
  # Skip SMS outreach
  python all_rounder_2.py --skip-sms
  
  # Run scraping and qualification only
  python all_rounder_2.py --skip-salesforce --skip-sms
  
  # Run in headless mode
  python all_rounder_2.py --headless
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


def run_pipeline_from_payload(payload: Dict[str, Any], stop_check: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """Configure and execute the pipeline based on an HTTP payload.

    Args:
        payload: Dictionary of controls accepted from API clients. Supported keys:
            - "skip_scraping" (bool): Skip the Craigslist scraping step.
            - "skip_salesforce" (bool): Skip pushing qualified leads to Salesforce.
            - "skip_sms" (bool): Skip Twilio outreach for qualified leads.
            - "headless" (bool): Run Selenium in headless mode.
            - "url" (str): Override the default Craigslist search URL.
            - "initial_message" (str): Override the first SMS sent to leads.
        stop_check: Optional callback function that returns True if pipeline should stop.
                    Called periodically during execution to check for stop requests.

    Returns:
        Execution summary dictionary from :py:meth:`LeadGenerationPipeline.run`.

    Example:
        >>> payload = {"skip_sms": True, "headless": True}
        >>> summary = run_pipeline_from_payload(payload)
        >>> summary["status"]
        'success'
    """

    skip_scraping = bool(payload.get("skip_scraping", False))
    skip_salesforce = bool(payload.get("skip_salesforce", False))
    skip_sms = bool(payload.get("skip_sms", False))

    headless = payload.get("headless")
    if headless is not None:
        PipelineConfig.HEADLESS_MODE = bool(headless)
    else:
        PipelineConfig.HEADLESS_MODE = False

    url_override = payload.get("url")
    if url_override:
        PipelineConfig.TARGET_URL = url_override

    initial_message_override = payload.get("initial_message")
    if initial_message_override:
        PipelineConfig.INITIAL_MESSAGE = initial_message_override

    pipeline = LeadGenerationPipeline()
    return pipeline.run(
        skip_scraping=skip_scraping,
        skip_salesforce=skip_salesforce,
        skip_sms=skip_sms,
        stop_check=stop_check,
    )


if __name__ == "__main__":
    exit(main())