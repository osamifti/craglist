import time
import os
import csv
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager
import logging


class BakersfieldCraigslistScraper:
    """
    Complete scraper for Bakersfield Craigslist vehicle listings with detailed page scraping.
    Navigates to specific URL, applies filters, and scrapes individual vehicle details.
    """
    
    def __init__(self, headless: bool = False):
        """
        Initialize the scraper with embedded configuration.
        
        Args:
            headless: Whether to run browser in headless mode
        """
        self.headless = headless
        self.driver = None
        self.wait = None
        self._clicked_show_contact_recently = False
        
        # Target URL for Bakersfield Craigslist
        self.target_url = "https://losangeles.craigslist.org/search/cta?bundleDuplicates=1&purveyor=owner#search=2~gallery~0"
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # XPath and CSS selectors for the new scraping logic
        self.owner_filter_xpath = "//div[@class='cl-segmented-selector']//button[span[text()='owner']]"
        self.bundle_duplicates_css = "input[name='bundleDuplicates']"
        self.gallery_card_xpath = "//div[@class='gallery-card']"
        
        # XPath selectors for individual vehicle page scraping
        self.title_xpath = "//span[@id='titletextonly']"
        self.price_xpath = "//span[@class='price']"
        self.mileage_xpath = "//div[@class='attr auto_miles']/span[@class='valu']"
        self.vin_xpath = "//div[@class='attr auto_vin']/span[@class='valu']"
        self.vehicle_details_xpath = "//section[@id='postingbody']"
        self.show_contact_xpath = "//a[@class='show-contact']"
        
        # CSV filename
        self.csv_filename = f"bakersfield_craigslist_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Initialize Chrome driver
        self._setup_chrome_driver()
        self._setup_csv_file()
    
    def _setup_chrome_driver(self):
        """
        Setup Chrome WebDriver with automatic driver management.
        """
        try:
            self.logger.info("Setting up Chrome WebDriver...")
            
            # Chrome options for better scraping performance
            chrome_options = Options()
            
            if self.headless:
                chrome_options.add_argument("--headless=new")
                self.logger.info("Running in headless mode")
            
            # Additional options for stability
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            
            # Automatically download and setup ChromeDriver
            service = Service(ChromeDriverManager().install())
            
            # Initialize the driver
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.wait = WebDriverWait(self.driver, 15)
            
            self.logger.info("Chrome WebDriver initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Chrome WebDriver: {str(e)}")
            raise
    
    def _setup_csv_file(self):
        """
        Create CSV file with headers.
        """
        try:
            with open(self.csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Title', 'Price', 'Mileage', 'VIN', 'Vehicle Details', 'Scraped At'])
            self.logger.info(f"CSV file created: {self.csv_filename}")
        except Exception as e:
            self.logger.error(f"Error creating CSV file: {str(e)}")
            raise
    
    def _append_to_csv(self, title, price, mileage, vin, vehicle_details):
        """
        Append scraped data to CSV file.
        
        Args:
            title: Vehicle title
            price: Vehicle price
            mileage: Vehicle mileage
            vin: Vehicle VIN number
            vehicle_details: Complete vehicle details
        """
        try:
            with open(self.csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([title, price, mileage, vin, vehicle_details, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        except Exception as e:
            self.logger.error(f"Error appending to CSV: {str(e)}")
    
    def navigate_to_target_url(self):
        """
        Navigate to the target URL and wait for 40 seconds.
        
        Returns:
            bool: True if navigation successful, False otherwise
        """
        try:
            self.logger.info(f"Navigating to target URL: {self.target_url}")
            self.driver.get(self.target_url)
            
            self.logger.info("Waiting for 40 seconds for page to fully load...")
            time.sleep(40)
            
            self.logger.info("40-second wait completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error navigating to target URL: {str(e)}")
            return False
    
    def click_owner_filter(self):
        """
        Click on the owner filter button using XPath.
        
        Returns:
            bool: True if button clicked successfully, False otherwise
        """
        try:
            self.logger.info("Looking for owner filter button...")
            
            # Wait for the element to be clickable
            owner_button = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, self.owner_filter_xpath))
            )
            
            # Scroll to the element to ensure it's visible
            self.driver.execute_script("arguments[0].scrollIntoView(true);", owner_button)
            time.sleep(2)
            
            # Click the button
            owner_button.click()
            self.logger.info("Owner filter button clicked successfully")
            
            return True
            
        except TimeoutException:
            self.logger.error("Owner filter button not found or not clickable")
            return False
        except Exception as e:
            self.logger.error(f"Error clicking owner filter button: {str(e)}")
            return False
    
    def click_bundle_duplicates(self):
        """
        Click on the bundleDuplicates checkbox using CSS selector.
        
        Returns:
            bool: True if checkbox clicked successfully, False otherwise
        """
        try:
            self.logger.info("Looking for bundleDuplicates checkbox...")
            
            # Wait for the element to be clickable
            bundle_checkbox = self.wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, self.bundle_duplicates_css))
            )
            
            # Scroll to the element to ensure it's visible
            self.driver.execute_script("arguments[0].scrollIntoView(true);", bundle_checkbox)
            time.sleep(2)
            
            # Click the checkbox
            bundle_checkbox.click()
            self.logger.info("BundleDuplicates checkbox clicked successfully")
            
            return True
            
        except TimeoutException:
            self.logger.error("BundleDuplicates checkbox not found or not clickable")
            return False
        except Exception as e:
            self.logger.error(f"Error clicking bundleDuplicates checkbox: {str(e)}")
            return False
    
    def wait_after_buttons(self):
        """
        Wait for 5 seconds after both buttons are clicked.
        """
        self.logger.info("Waiting for 5 seconds after button clicks...")
        time.sleep(5)
        self.logger.info("5-second wait completed")
    
    def get_gallery_cards(self):
        """
        Get all gallery cards on the current page.
        
        Returns:
            List of WebElement objects representing gallery cards
        """
        try:
            self.logger.info("Looking for gallery cards...")
            
            # Wait for gallery cards to be present
            gallery_cards = self.wait.until(
                EC.presence_of_all_elements_located((By.XPATH, self.gallery_card_xpath))
            )
            
            self.logger.info(f"Found {len(gallery_cards)} gallery cards")
            return gallery_cards
            
        except TimeoutException:
            self.logger.error("No gallery cards found on the page")
            return []
        except Exception as e:
            self.logger.error(f"Error finding gallery cards: {str(e)}")
            return []
    
    def click_gallery_card(self, card_element):
        """
        Click on a specific gallery card.
        
        Args:
            card_element: WebElement representing the gallery card
        
        Returns:
            bool: True if card clicked successfully, False otherwise
        """
        try:
            # Scroll to the card to ensure it's visible
            self.driver.execute_script("arguments[0].scrollIntoView(true);", card_element)
            time.sleep(1)

            try:
                card_element.click()
            except Exception:
                # Fallback to JS click if regular click fails
                self.driver.execute_script("arguments[0].click();", card_element)

            self.logger.info("Gallery card clicked successfully")

            # Prefer explicit wait for a known detail element
            try:
                self.wait.until(
                    EC.presence_of_element_located((By.XPATH, self.title_xpath))
                )
            except TimeoutException:
                # As a fallback, wait for the posting body
                self.wait.until(
                    EC.presence_of_element_located((By.XPATH, self.vehicle_details_xpath))
                )

            return True

        except Exception as e:
            self.logger.error(f"Error clicking gallery card: {str(e)}")
            return False
    
    def click_show_contact(self):
        """
        Click on the show-contact button twice with 5-second wait between clicks to reveal contact information.
        
        Returns:
            bool: True if button clicked successfully, False otherwise
        """
        try:
            self.logger.info("Looking for show-contact button...")
            
            # Try to find the show-contact button
            try:
                show_contact_button = self.driver.find_element(By.XPATH, self.show_contact_xpath)
                
                # Check if the button is visible and clickable
                if show_contact_button.is_displayed() and show_contact_button.is_enabled():
                    # Scroll to the button to ensure it's visible
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", show_contact_button)
                    time.sleep(1)
                    
                    # Click the button first time
                    show_contact_button.click()
                    self.logger.info("Show-contact button clicked first time")
                    self._clicked_show_contact_recently = True
                    
                    # Wait for 5 seconds after first click
                    self.logger.info("Waiting for 5 seconds after first click...")
                    time.sleep(5)
                    
                    # Click the button second time
                    show_contact_button.click()
                    self.logger.info("Show-contact button clicked second time")
                    
                    # Wait for 2 seconds after second click
                    self.logger.info("Waiting for 2 seconds after second click...")
                    time.sleep(2)
                    
                    self.logger.info("Contact information should now be visible - proceeding with scraping")
                    return True
                else:
                    self.logger.info("Show-contact button found but not clickable (may already be expanded)")
                    return True  # Return True as the contact info might already be visible
                    
            except NoSuchElementException:
                self.logger.info("Show-contact button not found (contact info may already be visible)")
                return True  # Return True as the contact info might already be visible
                
        except Exception as e:
            self.logger.warning(f"Error clicking show-contact button: {str(e)}")
            return True  # Continue with scraping even if this fails
        
    def scrape_vehicle_details(self):
        """
        Scrape detailed information from the current vehicle page.
        First clicks show-contact button to reveal contact information.
        
        Returns:
            dict: Dictionary containing scraped vehicle details
        """
        try:
            self.logger.info("Scraping vehicle details from individual page...")
            
            # First, try to click the show-contact button to reveal contact information
            self.click_show_contact()
            
            # Initialize default values
            title = "Not Found"
            price = "Not Found"
            mileage = "Not Found"
            vin = "Not Found"
            vehicle_details = "Not Found"
            
            # Scrape title
            try:
                title_element = self.driver.find_element(By.XPATH, self.title_xpath)
                title = title_element.text.strip()
                self.logger.info(f"Title scraped: {title}")
            except NoSuchElementException:
                self.logger.warning("Title not found on this page")
            except Exception as e:
                self.logger.warning(f"Error scraping title: {str(e)}")
            
            # Scrape price
            try:
                price_element = self.driver.find_element(By.XPATH, self.price_xpath)
                price = price_element.text.strip()
                self.logger.info(f"Price scraped: {price}")
            except NoSuchElementException:
                self.logger.warning("Price not found on this page")
            except Exception as e:
                self.logger.warning(f"Error scraping price: {str(e)}")
            
            # Scrape mileage
            try:
                mileage_element = self.driver.find_element(By.XPATH, self.mileage_xpath)
                mileage = mileage_element.text.strip()
                self.logger.info(f"Mileage scraped: {mileage}")
            except NoSuchElementException:
                self.logger.warning("Mileage not found on this page")
            except Exception as e:
                self.logger.warning(f"Error scraping mileage: {str(e)}")
            
            # Scrape VIN
            try:
                vin_element = self.driver.find_element(By.XPATH, self.vin_xpath)
                vin = vin_element.text.strip()
                self.logger.info(f"VIN scraped: {vin}")
            except NoSuchElementException:
                self.logger.warning("VIN not found on this page")
            except Exception as e:
                self.logger.warning(f"Error scraping VIN: {str(e)}")
            
            # Scrape vehicle details
            try:
                details_element = self.driver.find_element(By.XPATH, self.vehicle_details_xpath)
                vehicle_details = details_element.text.strip()
                self.logger.info(f"Vehicle details scraped: {len(vehicle_details)} characters")
            except NoSuchElementException:
                self.logger.warning("Vehicle details not found on this page")
            except Exception as e:
                self.logger.warning(f"Error scraping vehicle details: {str(e)}")
            
            return {
                'title': title,
                'price': price,
                'mileage': mileage,
                'vin': vin,
                'vehicle_details': vehicle_details
            }
            
        except Exception as e:
            self.logger.error(f"Error scraping vehicle details: {str(e)}")
            return {
                'title': "Not Found",
                'price': "Not Found",
                'mileage': "Not Found",
                'vin': "Not Found",
                'vehicle_details': "Not Found"
            }
    
    def go_back_to_gallery(self):
        """
        Navigate back to the gallery page.
        
        Returns:
            bool: True if navigation back successful, False otherwise
        """
        try:
            self.logger.info("Navigating back to gallery page...")
            self.driver.back()

            # If show-contact was clicked, Craigslist opens a modal/overlay history entry.
            # Perform a second back to return to the gallery.
            if self._clicked_show_contact_recently:
                self.logger.info("Detected show-contact click earlier; performing an additional back navigation...")
                time.sleep(0.5)
                self.driver.back()
                # reset flag after double back
                self._clicked_show_contact_recently = False

            # Wait for gallery cards to be present again (after one or two backs)
            self.wait.until(
                EC.presence_of_all_elements_located((By.XPATH, self.gallery_card_xpath))
            )

            self.logger.info("Successfully navigated back to gallery")
            return True

        except Exception as e:
            self.logger.error(f"Error navigating back to gallery: {str(e)}")
            return False
    
    def run_scraping_process(self):
        """
        Run the complete scraping process as specified.
            
        Returns:
            bool: True if scraping completed successfully, False otherwise
        """
        try:
            self.logger.info("Starting complete scraping process...")
            
            # Step 1: Navigate to target URL and wait 40 seconds
            if not self.navigate_to_target_url():
                self.logger.error("Failed to navigate to target URL")
                return False
            
            # Step 2: Click owner filter button
            if not self.click_owner_filter():
                self.logger.error("Failed to click owner filter button")
                return False
            
            # Step 3: Click bundleDuplicates checkbox
            if not self.click_bundle_duplicates():
                self.logger.error("Failed to click bundleDuplicates checkbox")
                return False
            
            # Step 4: Wait 5 seconds after both buttons are clicked
            self.wait_after_buttons()
            
            # Step 5: Process gallery cards with re-fetch after navigation
            processed_cards = 0
            card_index = 0

            # Initial discovery of cards
            gallery_cards = self.get_gallery_cards()
            if not gallery_cards:
                self.logger.error("No gallery cards found")
                return False

            total_cards_on_page = len(gallery_cards)
            self.logger.info(f"Found {total_cards_on_page} gallery cards to process")

            while True:
                try:
                    # If we've processed all cards, stop
                    if card_index >= total_cards_on_page:
                        break

                    self.logger.info(f"Processing gallery card {card_index + 1}/{total_cards_on_page}")

                    # Re-fetch cards to avoid stale references after navigation
                    try:
                        current_cards = self.get_gallery_cards()
                        if not current_cards or card_index >= len(current_cards):
                            self.logger.warning("Cards list changed or not available; stopping processing loop.")
                            break
                        card = current_cards[card_index]
                    except StaleElementReferenceException:
                        self.logger.warning("Stale element list; re-fetching and continuing.")
                        continue

                    # Click the gallery card
                    if not self.click_gallery_card(card):
                        self.logger.warning(f"Failed to click gallery card {card_index + 1}, skipping...")
                        card_index += 1
                        continue

                    # Scrape vehicle details
                    vehicle_data = self.scrape_vehicle_details()

                    # Append to CSV
                    self._append_to_csv(
                        vehicle_data['title'],
                        vehicle_data['price'],
                        vehicle_data['mileage'],
                        vehicle_data['vin'],
                        vehicle_data['vehicle_details']
                    )

                    processed_cards += 1
                    self.logger.info(f"Successfully processed card {card_index + 1}")

                    # Go back to gallery for next card
                    if not self.go_back_to_gallery():
                        self.logger.warning("Failed to go back to gallery, stopping...")
                        break

                    # Move to next card
                    card_index += 1

                except Exception as e:
                    self.logger.error(f"Error processing gallery card {card_index + 1}: {str(e)}")
                    card_index += 1
                    continue
            
            self.logger.info(f"Scraping process completed! Processed {processed_cards} cards")
            self.logger.info(f"Data saved to: {self.csv_filename}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error during scraping process: {str(e)}")
            return False
    
    def close(self):
        """Close the browser and cleanup resources."""
        if self.driver:
            self.driver.quit()
            self.logger.info("Browser closed successfully")


def main():
    """
    Main function to run the Bakersfield Craigslist scraper.
    """
    print("Bakersfield Craigslist Detailed Vehicle Scraper")
    print("=" * 60)
    print("This scraper will:")
    print("1. Navigate to the target URL")
    print("2. Wait for 40 seconds")
    print("3. Click owner filter button")
    print("4. Click bundleDuplicates checkbox")
    print("5. Wait for 5 seconds")
    print("6. Click each gallery card and scrape detailed information")
    print("7. Click show-contact button first time")
    print("8. Wait for 5 seconds after first click")
    print("9. Click show-contact button second time")
    print("10. Wait for 2 seconds after second click")
    print("11. Scrape vehicle details (title, price, mileage, VIN, details)")
    print("12. Save all data to CSV file")
    print("=" * 60)
    
    scraper = None
    try:
        # Ask user for headless mode preference
        headless_choice = input("\nRun in headless mode? (y/n): ").strip().lower()
        headless_mode = headless_choice in ['y', 'yes']
        
        print(f"\nInitializing scraper (headless: {headless_mode})...")
        scraper = BakersfieldCraigslistScraper(headless=headless_mode)
        
        print("Starting detailed scraping process...")
        success = scraper.run_scraping_process()
        
        if success:
            print(f"\nScraping completed successfully!")
            print(f"Data saved to: {scraper.csv_filename}")
        else:
            print("\nScraping failed. Check the logs for details.")
    
    except KeyboardInterrupt:
        print("\nScraping interrupted by user.")
    except Exception as e:
        print(f"Error during scraping: {str(e)}")
    
    finally:
        if scraper:
            scraper.close()
            print("Browser closed.")


if __name__ == "__main__":
    main()