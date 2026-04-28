import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import json
import os

# Coin isimlerini Yahoo Finance sembollerine dönüştüren eşleme tablosu
COIN_TO_YAHOO_SYMBOL = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "BNB": "BNB-USD",
    "Solana": "SOL-USD",
    "XRP": "XRP-USD",
    "Dogecoin": "DOGE-USD",
    "Toncoin": "TON-USD",
    "Cardano": "ADA-USD",
    "Shiba Inu": "SHIB-USD",
    "Avalanche": "AVAX-USD",
    "Polkadot": "DOT-USD",
    "TRON": "TRX-USD",
    "Chainlink": "LINK-USD",
    "Polygon": "MATIC-USD",
    "Internet Computer": "ICP-USD",
    "Litecoin": "LTC-USD",
    "Uniswap": "UNI-USD",
    "Bitcoin Cash": "BCH-USD"
}

# Filtrelenecek terimler - Daha güvenli ve spesifik liste
FILTER_TERMS = {
    # Wrapped/Bridged versions
    "Wrapped Bitcoin", "Wrapped Ethereum", "Wrapped BNB", "Wrapped", "Bridged",
    "Wormhole", "Portal Bitcoin", "Portal Ethereum", "Portal",
    
    # Stablecoins (tam isimlerle)
    "Tether", "USD Coin", "Dai Stablecoin", "First Digital USD", "TrueUSD", 
    "Pax Dollar", "USDD", "Frax", "Binance USD", "Gemini Dollar", "PayPal USD",
    
    # Fiat currencies  
    "Euro Coin", "JPY Coin", "GBP Coin", "AUD Coin", "TerraUSD",
    
    # Specific blockchain versions
    "Binance-Peg", "BSC Token", "Ethereum Classic", 
    
    # Test/Beta tokens
    "Testnet", "Test Token", "Beta", 
    
    # Pool/Vault tokens
    "Liquidity Pool", "Vault Token", "Receipt Token", "IOU Token",
    
    # Staked versions (tam isimler)
    "Staked Ether", "Staked Ethereum", "Staked BNB", "Staked Solana",
    "Staked Cardano", "Staked Polkadot", "Staked Cosmos", "Staked",
    
    # Tokenized versions
    "Tokenized Bitcoin", "Tokenized Ethereum", "Tokenized"
}

# Bilinen Yahoo Finance eşlemeleri
YF_MAPPING = {
    'TETHER-USD': 'USDT-USD',
    'STELLAR-USD': 'XLM-USD',
    'CURVE-USD': 'CRV-USD',
    'HEDERA-USD': 'HBAR-USD',
    'ZCASH-USD': 'ZEC-USD',
    'INJECTIVE-USD': 'INJ-USD',
    'FILECOIN-USD': 'FIL-USD',
    'MONERO-USD': 'XMR-USD',
    'COSMOS-USD': 'ATOM-USD',
    'ALGORAND-USD': 'ALGO-USD',
    'THORCHAIN-USD': 'RUNE-USD',
    'OPTIMISM-USD': 'OP-USD',
    'ARBITRUM-USD': 'ARB-USD',
    'SPARK-USD': 'FLR-USD',
    'PEPE-USD': 'PEPE-USD',
    'CHAINLINK-USD': 'LINK-USD',
    'INTERNET-USD': 'ICP-USD',
    'NEAR-USD': 'NEAR-USD',
    'APTOS-USD': 'APT-USD',
    'ETHEREUM-USD': 'ETH-USD',
    'BITCOIN-USD': 'BTC-USD'
}

class ThreadSafeCounter:
    """Thread-safe sayaç ve rate limiter"""
    def __init__(self):
        self.lock = threading.Lock()
        self.processed = 0
        self.successful = 0
        self.failed = 0
        self.rate_limited = 0
        self.last_request_times = []
        
    def increment_processed(self):
        with self.lock:
            self.processed += 1
            
    def increment_successful(self):
        with self.lock:
            self.successful += 1
            
    def increment_failed(self):
        with self.lock:
            self.failed += 1
            
    def increment_rate_limited(self):
        with self.lock:
            self.rate_limited += 1
    
    def get_stats(self):
        with self.lock:
            return {
                'processed': self.processed,
                'successful': self.successful,
                'failed': self.failed,
                'rate_limited': self.rate_limited
            }
    
    def should_wait(self, max_requests_per_minute=20):
        """Rate limiting kontrolü"""
        with self.lock:
            now = time.time()
            # Son 1 dakikadaki istekleri temizle
            self.last_request_times = [t for t in self.last_request_times if now - t < 60]
            
            if len(self.last_request_times) >= max_requests_per_minute:
                return True
            
            self.last_request_times.append(now)
            return False

class CacheManager:
    """Sonuçları önbelleğe alan ve kaydeden sınıf"""
    def __init__(self, cache_file="yahoo_symbol_cache.json"):
        self.cache_file = cache_file
        self.cache = self.load_cache()
        self.lock = threading.Lock()
    
    def load_cache(self):
        """Önbelleği dosyadan yükle"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_cache(self):
        """Önbelleği dosyaya kaydet"""
        with self.lock:
            try:
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.cache, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Cache kaydetme hatası: {e}")
    
    def get(self, key):
        """Cache'den değer al"""
        with self.lock:
            return self.cache.get(key)
    
    def set(self, key, value):
        """Cache'e değer ekle"""
        with self.lock:
            self.cache[key] = value

def is_real_coin(coin_name):
    """Filtre listesindeki terimleri içermeyen gerçek coin'leri kontrol eder"""
    if not coin_name:
        return False
        
    coin_name_clean = coin_name.strip()
    
    # Debug: Önemli coinlerin filtrelenmesini önle
    important_coins = ["CARDANO", "BITCOIN", "ETHEREUM", "SOLANA", "XRP", "DOGECOIN", 
                      "BNB", "TONCOIN", "AVALANCHE", "POLKADOT", "TRON", "CHAINLINK",
                      "POLYGON", "LITECOIN", "UNISWAP", "BITCOIN CASH", "SHIBA INU",
                      "INTERNET COMPUTER", "NEAR PROTOCOL", "APTOS", "ARBITRUM",
                      "OPTIMISM", "MANTLE", "FILECOIN", "HEDERA", "COSMOS"]
    
    if any(important in coin_name_clean.upper() for important in important_coins):
        return True
    
    # Tam eşleşme kontrolü (daha güvenli)
    for term in FILTER_TERMS:
        if term.upper() == coin_name_clean.upper():  # Tam eşleşme
            return False
        if term.upper() in coin_name_clean.upper() and len(term) > 3:  # Uzun terimler için kısmi eşleşme
            return False
    
    # Stablecoin kontrolü (daha spesifik)
    stablecoin_patterns = [
        " USD", "USD ", "(USD)", "-USD-", 
        " EUR", "EUR ", "(EUR)",
        " JPY", "JPY ", "(JPY)"
    ]
    
    for pattern in stablecoin_patterns:
        if pattern in coin_name_clean.upper():
            return False
    
    return True

def get_real_top_coins(limit=300):
    """CoinGecko API'sinden gerçek coin listesi alır - Çoklu sayfa desteği"""
    all_coins = []
    page = 1
    per_page = 100  # CoinGecko maksimum per_page değeri
    max_attempts = 5
    
    print(f"🔍 {limit} coin için CoinGecko API'den veri çekiliyor...")
    
    while len(all_coins) < limit and page <= 10:  # Maksimum 10 sayfa kontrol et
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",  # Market cap sıralaması kullan
            "per_page": per_page,
            "page": page,
            "sparkline": "false",
            "locale": "en"
        }

        print(f"  📄 Sayfa {page} çekiliyor... (Hedef: {limit - len(all_coins)} coin kaldı)")
        
        for attempt in range(max_attempts):
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                response = requests.get(url, params=params, headers=headers, timeout=30)
                
                if response.status_code == 429:
                    wait_time = 65 + (attempt * 10)
                    print(f"    ⏳ Rate limit (429). {wait_time} saniye bekleniyor... (Deneme {attempt + 1})")
                    time.sleep(wait_time)
                    continue
                    
                if response.status_code != 200:
                    print(f"    ❌ HTTP {response.status_code} hatası. Tekrar deneniyor...")
                    time.sleep(5 * (attempt + 1))
                    continue
                    
                data = response.json()
                
                if not data:
                    print(f"    ⚠️  Sayfa {page} boş, sonlandırılıyor.")
                    break
                
                # DEBUG: İlk birkaç coin'i kontrol et
                if page == 1:
                    print("    🔍 İlk 10 coin (debug):")
                    for i, coin in enumerate(data[:10]):
                        coin_name = coin.get("name", "")
                        market_cap_rank = coin.get("market_cap_rank", "N/A")
                        is_valid = is_real_coin(coin_name)
                        print(f"      {market_cap_rank:2}. {coin_name} -> {'✅' if is_valid else '❌'}")
                
                # Filtreleme uygula
                page_coins = []
                for coin in data:
                    coin_name = coin.get("name", "")
                    market_cap_rank = coin.get("market_cap_rank", None)
                    
                    if coin_name and is_real_coin(coin_name):
                        # Ek kontroller
                        if not coin_name.startswith(("Wrapped ", "Staked ", "Bridged ")):
                            page_coins.append({
                                'name': coin_name,
                                'rank': market_cap_rank,
                                'symbol': coin.get('symbol', '').upper()
                            })
                
                all_coins.extend([coin['name'] for coin in page_coins])
                print(f"    ✅ Sayfa {page}: {len(page_coins)} geçerli coin bulundu (Toplam: {len(all_coins)})")
                
                # Debug: Cardano var mı kontrol et
                cardano_found = any(coin['name'].upper() == 'CARDANO' for coin in page_coins)
                if cardano_found:
                    print(f"    🎯 CARDANO bulundu! Sayfa {page}")
                
                # Hedef coin sayısına ulaştık mı?
                if len(all_coins) >= limit:
                    print(f"    🎯 Hedef {limit} coin sayısına ulaşıldı!")
                    break
                
                # Sayfa arası nazik bekleme
                time.sleep(2)
                break
                
            except requests.exceptions.RequestException as e:
                print(f"    ❌ İstek hatası (Deneme {attempt + 1}): {e}")
                if attempt < max_attempts - 1:
                    time.sleep(10 * (attempt + 1))
                    continue
                else:
                    print(f"    💀 Sayfa {page} için tüm denemeler başarısız.")
                    break
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                print(f"    ❌ JSON parse hatası: {e}")
                break
        else:
            # For döngüsü break ile kırılmadıysa (tüm denemeler başarısız)
            break
        
        page += 1
    
    # Sonuçları sınırla ve döndür
    result_coins = all_coins[:limit]
    print(f"✅ Toplam {len(result_coins)} geçerli coin bulundu.")
    
    if len(result_coins) < limit:
        print(f"⚠️  Hedeflenen {limit} coin'den sadece {len(result_coins)} tanesi bulunabildi.")
    
    # İlk 15 coin'i göster (Cardano'yu bulmak için)
    if result_coins:
        print("📋 İlk 15 coin:")
        for i, coin in enumerate(result_coins[:15], 1):
            print(f"  {i:2d}. {coin}")
        
        # Cardano var mı kontrol et
        if 'Cardano' in result_coins:
            cardano_index = result_coins.index('Cardano') + 1
            print(f"🎯 CARDANO bulundu! Liste pozisyonu: {cardano_index}")
        else:
            print("❌ CARDANO bulunamadı!")
    
    return result_coins

def convert_to_yahoo_symbols(coin_names):
    """Coin isimlerini Yahoo Finance sembollerine dönüştürür"""
    yahoo_symbols = []
    for name in coin_names:
        symbol = COIN_TO_YAHOO_SYMBOL.get(name, f"{name.split()[0].upper()}-USD")
        yahoo_symbols.append(symbol)
    return yahoo_symbols

def yf_best_match_worker(symbol_data, counter, cache_manager):
    """Thread worker fonksiyonu"""
    symbol, thread_id = symbol_data
    
    # Cache kontrolü
    cached_result = cache_manager.get(symbol)
    if cached_result:
        counter.increment_processed()
        counter.increment_successful()
        print(f"[T{thread_id}] [Cache] {symbol} -> {cached_result}")
        return cached_result
    
    # Bilinen eşlemeler kontrolü
    if symbol in YF_MAPPING:
        result = YF_MAPPING[symbol]
        cache_manager.set(symbol, result)
        counter.increment_processed()
        counter.increment_successful()
        print(f"[T{thread_id}] [Known] {symbol} -> {result}")
        return result
    
    # Rate limiting kontrolü
    if counter.should_wait():
        wait_time = 60 + (thread_id * 2)  # Thread ID'ye göre farklı bekleme
        print(f"[T{thread_id}] Rate limit - {wait_time} saniye bekleniyor...")
        time.sleep(wait_time)
    
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={symbol}"
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            headers = {
                'User-Agent': f'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (Thread-{thread_id})'
            }
            
            response = requests.get(url, timeout=15, headers=headers)
            
            if response.status_code == 429:
                counter.increment_rate_limited()
                wait_time = 30 * (attempt + 1) + (thread_id * 5)
                print(f"[T{thread_id}] [429] {symbol}: {wait_time}s bekleniyor (Deneme {attempt + 1})")
                time.sleep(wait_time)
                continue
            
            if response.status_code != 200:
                print(f"[T{thread_id}] [HTTP {response.status_code}] {symbol}")
                if attempt == max_retries - 1:
                    counter.increment_failed()
                    return symbol
                time.sleep(5 + thread_id)
                continue
                
            data = response.json()
            quotes = data.get("quotes", [])
            
            if quotes and len(quotes) > 0:
                found_symbol = quotes[0].get("symbol")
                if found_symbol:
                    cache_manager.set(symbol, found_symbol)
                    counter.increment_processed()
                    counter.increment_successful()
                    print(f"[T{thread_id}] [OK] {symbol} -> {found_symbol}")
                    return found_symbol
            
            counter.increment_failed()
            print(f"[T{thread_id}] [Not Found] {symbol}")
            return symbol
            
        except requests.exceptions.Timeout:
            print(f"[T{thread_id}] [Timeout] {symbol} (Deneme {attempt + 1})")
            if attempt < max_retries - 1:
                time.sleep(10 + thread_id * 2)
                continue
        except Exception as e:
            print(f"[T{thread_id}] [Error] {symbol}: {e}")
            if attempt < max_retries - 1:
                time.sleep(5 + thread_id)
                continue
    
    counter.increment_failed()
    return symbol

def process_batch_parallel(symbols, batch_size=10, max_workers=3):
    """Batch'leri paralel olarak işler"""
    cache_manager = CacheManager()
    counter = ThreadSafeCounter()
    all_results = []
    original_to_found = {}  # Orijinal sembol -> Bulunan sembol mapping
    
    # Sembolleri batch'lere böl
    batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
    
    print(f"Toplam {len(symbols)} sembol, {len(batches)} batch'te işlenecek")
    print(f"Batch boyutu: {batch_size}, Max worker: {max_workers}")
    
    for batch_idx, batch in enumerate(batches, 1):
        print(f"\n=== BATCH {batch_idx}/{len(batches)} ({len(batch)} sembol) ===")
        
        # Her sembol için thread ID ekle
        symbol_data = [(symbol, i) for i, symbol in enumerate(batch)]
        batch_results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Future'ları submit et
            future_to_symbol = {
                executor.submit(yf_best_match_worker, data, counter, cache_manager): data[0]  
                for data in symbol_data
            }
            
            # Sonuçları topla
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    original_to_found[symbol] = result
                    batch_results.append(result)
                except Exception as exc:
                    print(f'[ERROR] {symbol} işlenirken hata: {exc}')
                    original_to_found[symbol] = symbol
                    batch_results.append(symbol)  # Hata varsa orijinal sembolü ekle
        
        all_results.extend(batch_results)
        
        # Batch arası bekleme
        if batch_idx < len(batches):
            wait_time = 10 + len(batch)  # Batch boyutuna göre bekleme
            print(f"Batch tamamlandı. {wait_time} saniye bekleniyor...")
            time.sleep(wait_time)
        
        # İstatistikleri göster
        stats = counter.get_stats()
        print(f"İstatistikler: İşlenen={stats['processed']}, Başarılı={stats['successful']}, "
              f"Başarısız={stats['failed']}, Rate Limited={stats['rate_limited']}")
    
    # Cache'i kaydet
    cache_manager.save_cache()
    
    # Sadece Yahoo Finance'te bulunan coinleri filtrele (orijinal != bulunan)
    found_symbols = []
    not_found_symbols = []
    
    for original, found in original_to_found.items():
        if original != found:  # Yahoo'da farklı bir sembol bulundu
            if found not in found_symbols:  # Duplicate kontrolü
                found_symbols.append(found)
        else:  # Yahoo'da bulunamadı (orijinal sembol döndü)
            not_found_symbols.append(original)
    
    print(f"\n📊 Yahoo Finance Kontrol Sonuçları:")
    print(f"   ✅ Yahoo'da bulunan: {len(found_symbols)}")
    print(f"   ❌ Yahoo'da bulunamayan: {len(not_found_symbols)}")
    
    return found_symbols, counter.get_stats(), not_found_symbols

def main():
    """Ana fonksiyon"""
    print("🚀 Hibrit Crypto Ticker Processor Başlatılıyor...")
    
    # Konfigürasyon
    BATCH_SIZE = 30        # Her batch'te kaç sembol
    MAX_WORKERS = 5       # Aynı anda kaç thread
    COIN_LIMIT = 80     # Kaç coin alınacak
    
    print(f"Konfigürasyon: Batch={BATCH_SIZE}, Workers={MAX_WORKERS}, Limit={COIN_LIMIT}")
    
    # Coin listesi al
    print("\n📊 Coin listesi alınıyor...")
    coin_names = get_real_top_coins(COIN_LIMIT)
    
    if not coin_names:
        print("❌ Coin listesi alınamadı!")
        return
    
    print(f"✅ {len(coin_names)} coin bulundu.")
    
    # Yahoo Finance sembollerine dönüştür
    print("\n🔄 Yahoo Finance sembollerine dönüştürülüyor...")
    yahoo_symbols = convert_to_yahoo_symbols(coin_names)
    
    print("İlk 10 dönüşüm:")
    for name, symbol in zip(coin_names[:10], yahoo_symbols[:10]):
        print(f"  {name.ljust(20)} -> {symbol}")
    
    # Paralel batch işleme
    print(f"\n⚡ {len(yahoo_symbols)} sembol paralel batch işleme ile kontrol ediliyor...")
    start_time = time.time()
    
    corrected_symbols, final_stats, not_found_symbols = process_batch_parallel(
        yahoo_symbols, 
        batch_size=BATCH_SIZE, 
        max_workers=MAX_WORKERS
    )
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Sonuçları yazdır
    print(f"\n🎉 === SONUÇLAR ===")
    print(f"⏱️  Toplam süre: {processing_time:.1f} saniye")
    print(f"📈 Yahoo'da bulunan sembol sayısı: {len(corrected_symbols)}")
    print(f"📊 İstatistikler:")
    print(f"   ✅ İşlenen: {final_stats['processed']}")
    print(f"   🎯 Başarılı: {final_stats['successful']}")
    print(f"   ❌ Başarısız: {final_stats['failed']}")
    print(f"   ⏳ Rate Limited: {final_stats['rate_limited']}")
    print(f"   🚀 Hız: {final_stats['processed']/processing_time:.2f} sembol/saniye")
    print(f"   📈 Başarı oranı: {final_stats['successful']/max(final_stats['processed'],1)*100:.1f}%")
    
    print(f"\n🎯 === SADECE YAHOO FINANCE'TE BULUNAN COİNLER ===")
    print(f"📋 Yahoo Finance ticker listesi ({len(corrected_symbols)} adet):")
    print(corrected_symbols)
    
    print(f"\n📄 CSV formatı:")
    print(",".join(corrected_symbols))
    
    # Bulunamayan coinleri göster (opsiyonel)
    if not_found_symbols:
        print(f"\n⚠️  Yahoo Finance'te bulunamayan coinler ({len(not_found_symbols)} adet):")
        print(", ".join(not_found_symbols[:20]))  # İlk 20'sini göster
        if len(not_found_symbols) > 20:
            print(f"... ve {len(not_found_symbols) - 20} coin daha")
    
    # Dosyaya kaydet
    output_file = "yahoo_symbols_result.txt"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=== YAHOO FINANCE'TE BULUNAN COİNLER ===\n")
            f.write("Ticker List:\n")
            f.write(",".join(corrected_symbols))
            f.write(f"\n\n=== İSTATİSTİKLER ===\n")
            f.write(f"Yahoo'da bulunan: {len(corrected_symbols)}\n")
            f.write(f"Yahoo'da bulunamayan: {len(not_found_symbols)}\n")
            f.write(f"Toplam işlenen: {final_stats['processed']}\n")
            f.write(f"İşlem süresi: {processing_time:.1f}s\n")
            f.write(f"Hız: {final_stats['processed']/processing_time:.2f} sembol/saniye\n")
            f.write(f"Başarı oranı: {final_stats['successful']/max(final_stats['processed'],1)*100:.1f}%\n")
            
            if not_found_symbols:
                f.write(f"\n=== YAHOO'DA BULUNAMAYAN COİNLER ===\n")
                f.write("\n".join(not_found_symbols))
                
        print(f"💾 Sonuçlar '{output_file}' dosyasına kaydedildi.")
    except Exception as e:
        print(f"❌ Dosya kaydetme hatası: {e}")

if __name__ == "__main__":
    main()
