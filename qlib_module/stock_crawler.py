"""
Stock Crawler Module
Crawls stock data from EastMoney API at runtime.
Based on the logic from script/foud_collector.py with bug fixes.
"""
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# EastMoney API URLs
STOCK_KLINE_URL = "http://push2his.eastmoney.com/api/qt/stock/kline/get"

# Market code mapping
MARKET_CODE_MAP = {
    'sh': 1,  # Shanghai
    'sz': 0,  # Shenzhen
    'bj': 0,  # Beijing (BSE)
}


class StockCrawler:
    """
    Stock data crawler that fetches data from EastMoney API.
    
    This crawler can fetch historical daily stock data for individual stocks
    given a stock code and date range.
    """
    
    DEFAULT_START_DATE = "2000-01-01"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Referer": "http://quote.eastmoney.com/",
    }
    
    def __init__(self, save_dir: Optional[str] = None, delay: float = 0.2):
        """
        Initialize the stock crawler.
        
        Args:
            save_dir: Directory to save the crawled data (optional)
            delay: Delay between API requests to avoid rate limiting
        """
        self.save_dir = Path(save_dir) if save_dir else None
        self.delay = delay
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def normalize_stock_code(stock_code: str) -> tuple:
        """
        Normalize stock code to (market_code, pure_code) format.
        
        Handles various input formats:
        - 600000 -> (1, 600000) [Shanghai]
        - 000001 -> (0, 000001) [Shenzhen]
        - sh600000 -> (1, 600000)
        - sz000001 -> (0, 000001)
        - 600000.SH -> (1, 600000)
        - 000001.SZ -> (0, 000001)
        - bj430017 -> (0, 430017) [Beijing/BSE]
        
        Args:
            stock_code: Stock code in various formats
            
        Returns:
            Tuple of (market_code, pure_code)
        """
        stock_code = stock_code.strip().upper()
        
        # Handle suffix format (600000.SH)
        if '.' in stock_code:
            code, market = stock_code.split('.')
            market = market.lower()
            if market in ['sh', 'ss']:
                return 1, code
            elif market in ['sz']:
                return 0, code
            elif market == 'bj':
                return 0, code  # BSE uses market code 0
        
        # Handle prefix format (sh600000, SH600000)
        stock_code_lower = stock_code.lower()
        if stock_code_lower.startswith('sh'):
            return 1, stock_code[2:]
        elif stock_code_lower.startswith('sz'):
            return 0, stock_code[2:]
        elif stock_code_lower.startswith('bj'):
            return 0, stock_code[2:]  # BSE uses market code 0
        
        # Pure code - determine market by code pattern
        code = stock_code
        if code.startswith('6'):  # Shanghai main board
            return 1, code
        elif code.startswith('68'):  # Shanghai STAR Market
            return 1, code
        elif code.startswith(('0', '3')):  # Shenzhen main board and ChiNext
            return 0, code
        elif code.startswith(('4', '8')):  # Beijing Stock Exchange (BSE)
            return 0, code
        else:
            # Default to Shanghai
            return 1, code
    
    @staticmethod
    def format_stock_code(stock_code: str, format_type: str = 'qlib') -> str:
        """
        Format stock code to specified format.
        
        Args:
            stock_code: Stock code
            format_type: 'qlib' for sh600000/sz000001 format, 
                        'suffix' for 600000.SH format
        
        Returns:
            Formatted stock code
        """
        market_code, code = StockCrawler.normalize_stock_code(stock_code)
        
        if format_type == 'qlib':
            prefix = 'sh' if market_code == 1 else 'sz'
            # Handle BSE stocks
            if code.startswith(('4', '8')) and len(code) == 6:
                prefix = 'bj'
            return f"{prefix}{code}"
        else:
            suffix = 'SH' if market_code == 1 else 'SZ'
            if code.startswith(('4', '8')) and len(code) == 6:
                suffix = 'BJ'
            return f"{code}.{suffix}"
    
    def _sleep(self):
        """Sleep to avoid rate limiting."""
        if self.delay > 0:
            time.sleep(self.delay)
    
    def fetch_stock_data(
        self,
        stock_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical stock data from EastMoney API.
        
        Args:
            stock_code: Stock code (various formats supported)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with columns: date, open, high, low, close, volume, amount
            or None if fetch fails
        """
        market_code, pure_code = self.normalize_stock_code(stock_code)
        
        # Set default dates
        if not start_date:
            start_date = self.DEFAULT_START_DATE
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Convert dates to API format (YYYYMMDD)
        start_str = start_date.replace("-", "")
        end_str = end_date.replace("-", "")
        
        # Build API request
        params = {
            "secid": f"{market_code}.{pure_code}",
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt": "101",  # 101 = daily
            "fqt": "1",    # 1 = forward adjusted (前复权)
            "beg": start_str,
            "end": end_str,
        }
        
        try:
            self._sleep()
            response = requests.get(
                STOCK_KLINE_URL,
                params=params,
                headers=self.HEADERS,
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch data for {stock_code}: HTTP {response.status_code}")
                return None
            
            data = response.json()
            
            if data.get("data") is None:
                logger.warning(f"No data available for {stock_code}")
                return None
            
            klines = data["data"].get("klines", [])
            if not klines:
                logger.warning(f"Empty klines for {stock_code}")
                return None
            
            # Parse klines data
            # Format: "date,open,close,high,low,volume,amount,amplitude,change_pct,change_amount,turnover"
            records = []
            for line in klines:
                parts = line.split(",")
                if len(parts) >= 7:
                    records.append({
                        "date": parts[0],
                        "open": float(parts[1]) if parts[1] != '-' else None,
                        "close": float(parts[2]) if parts[2] != '-' else None,
                        "high": float(parts[3]) if parts[3] != '-' else None,
                        "low": float(parts[4]) if parts[4] != '-' else None,
                        "volume": float(parts[5]) if parts[5] != '-' else None,
                        "amount": float(parts[6]) if parts[6] != '-' else None,
                    })
            
            df = pd.DataFrame(records)
            
            # Add symbol column
            df["symbol"] = self.format_stock_code(stock_code, format_type='qlib')
            
            # Reorder columns
            df = df[["date", "symbol", "open", "high", "low", "close", "volume", "amount"]]
            
            # Convert date to datetime
            df["date"] = pd.to_datetime(df["date"])
            
            # Sort by date
            df = df.sort_values("date").reset_index(drop=True)
            
            logger.info(f"Successfully fetched {len(df)} records for {stock_code}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {stock_code}: {e}")
            return None
    
    def fetch_multiple_stocks(
        self,
        stock_codes: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple stocks.
        
        Args:
            stock_codes: List of stock codes
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            progress_callback: Optional callback function(current, total, stock_code)
            
        Returns:
            Dictionary mapping stock codes to DataFrames
        """
        results = {}
        total = len(stock_codes)
        
        for i, code in enumerate(stock_codes):
            if progress_callback:
                progress_callback(i + 1, total, code)
            
            df = self.fetch_stock_data(code, start_date, end_date)
            if df is not None and not df.empty:
                formatted_code = self.format_stock_code(code, format_type='qlib')
                results[formatted_code] = df
        
        return results
    
    def save_to_csv(
        self,
        df: pd.DataFrame,
        stock_code: str,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Save DataFrame to CSV file.
        
        Args:
            df: DataFrame to save
            stock_code: Stock code (used for filename)
            output_dir: Output directory (uses self.save_dir if not specified)
            
        Returns:
            Path to saved file
        """
        save_path = Path(output_dir) if output_dir else self.save_dir
        if not save_path:
            raise ValueError("No save directory specified")
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        formatted_code = self.format_stock_code(stock_code, format_type='qlib')
        file_path = save_path / f"{formatted_code}.csv"
        
        df.to_csv(file_path, index=False)
        logger.info(f"Saved data to {file_path}")
        
        return str(file_path)
    
    def crawl_and_save(
        self,
        stock_codes: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Crawl data for multiple stocks and save to files.
        
        Args:
            stock_codes: List of stock codes
            start_date: Start date
            end_date: End date
            output_dir: Output directory
            
        Returns:
            Summary of crawl results
        """
        result = {
            "start_time": datetime.now().isoformat(),
            "stock_codes": stock_codes,
            "start_date": start_date,
            "end_date": end_date,
            "success_count": 0,
            "failed_count": 0,
            "failed_codes": [],
            "saved_files": [],
        }
        
        for code in stock_codes:
            try:
                df = self.fetch_stock_data(code, start_date, end_date)
                if df is not None and not df.empty:
                    file_path = self.save_to_csv(df, code, output_dir)
                    result["success_count"] += 1
                    result["saved_files"].append(file_path)
                else:
                    result["failed_count"] += 1
                    result["failed_codes"].append(code)
            except Exception as e:
                logger.error(f"Failed to process {code}: {e}")
                result["failed_count"] += 1
                result["failed_codes"].append(code)
        
        result["end_time"] = datetime.now().isoformat()
        result["status"] = "completed" if result["failed_count"] == 0 else "partial"
        
        return result


def crawl_stock_data(
    stock_codes: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    delay: float = 0.2,
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to crawl stock data.
    
    Args:
        stock_codes: List of stock codes
        start_date: Start date
        end_date: End date
        delay: Delay between requests
        
    Returns:
        Dictionary mapping stock codes to DataFrames
    """
    crawler = StockCrawler(delay=delay)
    return crawler.fetch_multiple_stocks(stock_codes, start_date, end_date)
