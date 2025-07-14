# 🔗 DeFi và Blockchain Trading

## 📝 Tổng Quan

Decentralized Finance (DeFi) đã tạo ra một paradigm shift trong trading. Với Total Value Locked (TVL) vượt 200 tỷ USD, DeFi mở ra những cơ hội arbitrage, yield farming, và algorithmic trading hoàn toàn mới. Blockchain technology cho phép transparent, trustless trading với smart contracts.

## 🎯 DeFi Ecosystem

### 1. Decentralized Exchanges (DEXs)
- **Uniswap**: Automated Market Maker (AMM) lớn nhất
- **SushiSwap**: Community-driven AMM
- **Curve**: Stablecoin và similar assets
- **Balancer**: Multi-token AMM với flexible weights

### 2. Lending Protocols
- **Aave**: Variable và stable interest rates
- **Compound**: Algorithmic money markets
- **MakerDAO**: Decentralized stablecoin (DAI)
- **Yearn Finance**: Yield optimization

### 3. Derivatives Protocols
- **dYdX**: Decentralized derivatives exchange
- **Perpetual Protocol**: Virtual AMM for perpetuals
- **Synthetix**: Synthetic assets
- **GMX**: Decentralized perpetual exchange

## 🤖 DeFi Trading Strategies

### Arbitrage Trading
```python
from web3 import Web3
import requests
import asyncio
from decimal import Decimal

class DeFiArbitrageBot:
    def __init__(self, web3_provider, private_key):
        self.w3 = Web3(Web3.HTTPProvider(web3_provider))
        self.private_key = private_key
        self.account = self.w3.eth.account.from_key(private_key)
        
        # DEX contracts
        self.uniswap_router = self.w3.eth.contract(
            address="0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
            abi=self.get_uniswap_abi()
        )
        
        self.sushiswap_router = self.w3.eth.contract(
            address="0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F",
            abi=self.get_sushiswap_abi()
        )
    
    async def find_arbitrage_opportunities(self, token_pairs):
        """
        Tìm arbitrage opportunities giữa các DEXs
        """
        opportunities = []
        
        for pair in token_pairs:
            token_a, token_b = pair
            
            # Lấy giá từ Uniswap
            uniswap_price = await self.get_uniswap_price(token_a, token_b)
            
            # Lấy giá từ SushiSwap
            sushiswap_price = await self.get_sushiswap_price(token_a, token_b)
            
            # Tính price difference
            price_diff = abs(uniswap_price - sushiswap_price)
            price_ratio = price_diff / min(uniswap_price, sushiswap_price)
            
            # Kiểm tra profit potential
            if price_ratio > 0.005:  # 0.5% threshold
                gas_cost = await self.estimate_gas_cost()
                profit = self.calculate_profit(
                    uniswap_price, sushiswap_price, gas_cost
                )
                
                if profit > 0:
                    opportunities.append({
                        'token_pair': pair,
                        'uniswap_price': uniswap_price,
                        'sushiswap_price': sushiswap_price,
                        'price_diff': price_diff,
                        'profit_potential': profit,
                        'buy_exchange': 'uniswap' if uniswap_price < sushiswap_price else 'sushiswap',
                        'sell_exchange': 'sushiswap' if uniswap_price < sushiswap_price else 'uniswap'
                    })
        
        return opportunities
    
    async def execute_arbitrage(self, opportunity, amount):
        """
        Thực hiện arbitrage trade
        """
        try:
            # Flash loan để không cần capital
            flash_loan_tx = await self.initiate_flash_loan(
                opportunity['token_pair'][0], amount
            )
            
            # Buy trên exchange rẻ hơn
            buy_tx = await self.execute_buy(
                opportunity['buy_exchange'],
                opportunity['token_pair'],
                amount
            )
            
            # Sell trên exchange đắt hơn
            sell_tx = await self.execute_sell(
                opportunity['sell_exchange'],
                opportunity['token_pair'],
                amount
            )
            
            # Repay flash loan
            repay_tx = await self.repay_flash_loan(flash_loan_tx)
            
            return {
                'success': True,
                'profit': self.calculate_actual_profit(buy_tx, sell_tx),
                'transactions': [flash_loan_tx, buy_tx, sell_tx, repay_tx]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_uniswap_price(self, token_a, token_b):
        """
        Lấy giá từ Uniswap
        """
        amount_in = self.w3.toWei(1, 'ether')  # 1 token
        
        amounts_out = self.uniswap_router.functions.getAmountsOut(
            amount_in, [token_a, token_b]
        ).call()
        
        return amounts_out[1] / amount_in
    
    async def monitor_mempool(self):
        """
        Monitor mempool cho MEV opportunities
        """
        def handle_pending_tx(tx_hash):
            tx = self.w3.eth.get_transaction(tx_hash)
            
            # Decode transaction
            if self.is_dex_transaction(tx):
                # Analyze for front-running opportunity
                opportunity = self.analyze_transaction(tx)
                
                if opportunity['profit_potential'] > 0:
                    # Execute front-running or back-running
                    self.execute_mev_strategy(opportunity)
        
        # Subscribe to pending transactions
        pending_filter = self.w3.eth.filter('pending')
        
        while True:
            for tx_hash in pending_filter.get_new_entries():
                handle_pending_tx(tx_hash)
            
            await asyncio.sleep(0.1)
```

### Yield Farming Strategy
```python
class YieldFarmingBot:
    def __init__(self, web3_provider, private_key):
        self.w3 = Web3(Web3.HTTPProvider(web3_provider))
        self.private_key = private_key
        self.account = self.w3.eth.account.from_key(private_key)
        
        # Protocol contracts
        self.protocols = {
            'aave': self.w3.eth.contract(
                address="0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9",
                abi=self.get_aave_abi()
            ),
            'compound': self.w3.eth.contract(
                address="0xc3d688B66703497DAA19211EEdff47f25384cdc3",
                abi=self.get_compound_abi()
            ),
            'yearn': self.w3.eth.contract(
                address="0x0bc529c00C6401aEF6D220BE8C6Ea1667F6Ad93e",
                abi=self.get_yearn_abi()
            )
        }
    
    async def find_best_yield_opportunities(self, assets):
        """
        Tìm yield opportunities tốt nhất
        """
        opportunities = []
        
        for asset in assets:
            # Lấy APY từ các protocols
            aave_apy = await self.get_aave_apy(asset)
            compound_apy = await self.get_compound_apy(asset)
            yearn_apy = await self.get_yearn_apy(asset)
            
            # Tính risk-adjusted return
            risk_scores = {
                'aave': 0.1,      # Low risk
                'compound': 0.15,  # Medium risk
                'yearn': 0.25     # Higher risk (auto-compounding)
            }
            
            risk_adjusted_returns = {
                'aave': aave_apy * (1 - risk_scores['aave']),
                'compound': compound_apy * (1 - risk_scores['compound']),
                'yearn': yearn_apy * (1 - risk_scores['yearn'])
            }
            
            # Tìm protocol tốt nhất
            best_protocol = max(risk_adjusted_returns, key=risk_adjusted_returns.get)
            
            opportunities.append({
                'asset': asset,
                'best_protocol': best_protocol,
                'expected_apy': risk_adjusted_returns[best_protocol],
                'raw_apy': {
                    'aave': aave_apy,
                    'compound': compound_apy,
                    'yearn': yearn_apy
                }
            })
        
        return opportunities
    
    async def execute_yield_farming(self, opportunity, amount):
        """
        Thực hiện yield farming
        """
        asset = opportunity['asset']
        protocol = opportunity['best_protocol']
        
        try:
            # Approve token
            approve_tx = await self.approve_token(asset, protocol, amount)
            
            # Deposit vào protocol
            deposit_tx = await self.deposit_to_protocol(protocol, asset, amount)
            
            # Monitor position
            position_id = await self.track_position(protocol, asset, amount)
            
            return {
                'success': True,
                'position_id': position_id,
                'expected_apy': opportunity['expected_apy'],
                'transactions': [approve_tx, deposit_tx]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def auto_compound(self, positions):
        """
        Tự động compound rewards
        """
        for position in positions:
            # Kiểm tra rewards
            rewards = await self.get_pending_rewards(position)
            
            if rewards > self.get_gas_threshold():
                # Claim rewards
                claim_tx = await self.claim_rewards(position)
                
                # Reinvest rewards
                reinvest_tx = await self.reinvest_rewards(position, rewards)
                
                print(f"Auto-compounded position {position['id']}: {rewards} tokens")
    
    async def rebalance_portfolio(self, current_positions):
        """
        Rebalance portfolio based on APY changes
        """
        for position in current_positions:
            # Lấy APY hiện tại
            current_apy = await self.get_current_apy(position['protocol'], position['asset'])
            
            # Tìm APY tốt hơn
            better_opportunities = await self.find_better_opportunities(
                position['asset'], current_apy
            )
            
            if better_opportunities:
                best_opportunity = max(better_opportunities, key=lambda x: x['apy'])
                
                # Kiểm tra có đáng di chuyển không (sau khi tính gas)
                if self.is_worth_moving(position, best_opportunity):
                    # Withdraw từ protocol cũ
                    await self.withdraw_from_protocol(position)
                    
                    # Deposit vào protocol mới
                    await self.deposit_to_protocol(
                        best_opportunity['protocol'],
                        position['asset'],
                        position['amount']
                    )
```

### Liquidity Mining Strategy
```python
class LiquidityMiningBot:
    def __init__(self, web3_provider, private_key):
        self.w3 = Web3(Web3.HTTPProvider(web3_provider))
        self.private_key = private_key
        self.account = self.w3.eth.account.from_key(private_key)
    
    async def find_liquidity_opportunities(self, token_pairs):
        """
        Tìm liquidity mining opportunities
        """
        opportunities = []
        
        for pair in token_pairs:
            token_a, token_b = pair
            
            # Lấy thông tin pool
            pool_info = await self.get_pool_info(token_a, token_b)
            
            # Tính APY từ fees
            fee_apy = await self.calculate_fee_apy(pool_info)
            
            # Tính APY từ liquidity mining rewards
            mining_apy = await self.calculate_mining_apy(pool_info)
            
            # Tính impermanent loss risk
            il_risk = await self.calculate_impermanent_loss_risk(token_a, token_b)
            
            # Total expected return
            total_apy = fee_apy + mining_apy
            risk_adjusted_apy = total_apy * (1 - il_risk)
            
            opportunities.append({
                'pair': pair,
                'pool_address': pool_info['address'],
                'fee_apy': fee_apy,
                'mining_apy': mining_apy,
                'total_apy': total_apy,
                'il_risk': il_risk,
                'risk_adjusted_apy': risk_adjusted_apy,
                'tvl': pool_info['tvl'],
                'volume_24h': pool_info['volume_24h']
            })
        
        # Sort by risk-adjusted APY
        opportunities.sort(key=lambda x: x['risk_adjusted_apy'], reverse=True)
        
        return opportunities
    
    async def provide_liquidity(self, opportunity, amount_a, amount_b):
        """
        Provide liquidity to pool
        """
        try:
            # Approve tokens
            approve_tx_a = await self.approve_token(
                opportunity['pair'][0], opportunity['pool_address'], amount_a
            )
            approve_tx_b = await self.approve_token(
                opportunity['pair'][1], opportunity['pool_address'], amount_b
            )
            
            # Add liquidity
            add_liquidity_tx = await self.add_liquidity(
                opportunity['pool_address'],
                amount_a, amount_b
            )
            
            # Stake LP tokens for mining rewards
            stake_tx = await self.stake_lp_tokens(
                opportunity['pool_address'],
                add_liquidity_tx['lp_tokens']
            )
            
            return {
                'success': True,
                'position_id': stake_tx['position_id'],
                'lp_tokens': add_liquidity_tx['lp_tokens'],
                'transactions': [approve_tx_a, approve_tx_b, add_liquidity_tx, stake_tx]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def monitor_impermanent_loss(self, positions):
        """
        Monitor impermanent loss cho liquidity positions
        """
        for position in positions:
            # Lấy current pool state
            current_state = await self.get_pool_state(position['pool_address'])
            
            # Tính impermanent loss
            il_percentage = await self.calculate_current_il(
                position['initial_state'], current_state
            )
            
            # Kiểm tra threshold
            if il_percentage > 0.05:  # 5% threshold
                # Consider removing liquidity
                decision = await self.should_remove_liquidity(position, il_percentage)
                
                if decision['should_remove']:
                    await self.remove_liquidity(position)
                    
                    # Optionally rebalance to different pool
                    if decision['rebalance_pool']:
                        await self.rebalance_to_pool(position, decision['new_pool'])
```

## 🔮 Advanced DeFi Strategies

### Flash Loan Arbitrage
```python
class FlashLoanArbitrage:
    def __init__(self, web3_provider, private_key):
        self.w3 = Web3(Web3.HTTPProvider(web3_provider))
        self.private_key = private_key
        
        # Flash loan providers
        self.aave_lending_pool = self.w3.eth.contract(
            address="0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9",
            abi=self.get_aave_lending_pool_abi()
        )
    
    async def execute_flash_loan_arbitrage(self, token, amount, arbitrage_path):
        """
        Thực hiện flash loan arbitrage
        """
        # Tạo smart contract call data
        call_data = self.encode_arbitrage_calls(arbitrage_path)
        
        # Execute flash loan
        tx = await self.aave_lending_pool.functions.flashLoan(
            [token],              # assets
            [amount],             # amounts
            [0],                  # modes (0 = no debt)
            self.account.address, # onBehalfOf
            call_data,            # params
            0                     # referralCode
        ).transact({
            'from': self.account.address,
            'gas': 2000000,
            'gasPrice': self.w3.toWei('20', 'gwei')
        })
        
        return tx
    
    def encode_arbitrage_calls(self, arbitrage_path):
        """
        Encode arbitrage calls for flash loan callback
        """
        # Example: Buy on Uniswap, sell on SushiSwap
        calls = []
        
        for step in arbitrage_path:
            if step['action'] == 'swap':
                call = self.encode_swap_call(
                    step['dex'],
                    step['token_in'],
                    step['token_out'],
                    step['amount']
                )
                calls.append(call)
        
        return self.w3.eth.abi.encode_abi(['bytes[]'], [calls])
```

### MEV (Maximal Extractable Value) Strategies
```python
class MEVBot:
    def __init__(self, web3_provider, private_key):
        self.w3 = Web3(Web3.HTTPProvider(web3_provider))
        self.private_key = private_key
        
        # MEV strategies
        self.strategies = {
            'front_running': self.front_running_strategy,
            'back_running': self.back_running_strategy,
            'sandwich': self.sandwich_strategy
        }
    
    async def monitor_mempool_for_mev(self):
        """
        Monitor mempool cho MEV opportunities
        """
        pending_filter = self.w3.eth.filter('pending')
        
        while True:
            for tx_hash in pending_filter.get_new_entries():
                tx = self.w3.eth.get_transaction(tx_hash)
                
                # Analyze transaction
                mev_opportunity = await self.analyze_transaction_for_mev(tx)
                
                if mev_opportunity:
                    # Execute MEV strategy
                    await self.execute_mev_strategy(mev_opportunity)
            
            await asyncio.sleep(0.1)
    
    async def front_running_strategy(self, target_tx):
        """
        Front-running strategy
        """
        # Decode target transaction
        decoded_tx = self.decode_dex_transaction(target_tx)
        
        if decoded_tx['action'] == 'large_buy':
            # Front-run with smaller buy
            front_run_tx = await self.create_front_run_transaction(
                decoded_tx['token'],
                decoded_tx['amount'] * 0.1,  # 10% of target amount
                target_tx['gasPrice'] + 1    # Higher gas price
            )
            
            return front_run_tx
    
    async def sandwich_strategy(self, target_tx):
        """
        Sandwich attack strategy
        """
        decoded_tx = self.decode_dex_transaction(target_tx)
        
        if decoded_tx['slippage_tolerance'] > 0.05:  # High slippage
            # Front-run transaction
            front_tx = await self.create_front_run_transaction(
                decoded_tx['token_in'],
                decoded_tx['amount_in'] * 0.5,
                target_tx['gasPrice'] + 1
            )
            
            # Back-run transaction
            back_tx = await self.create_back_run_transaction(
                decoded_tx['token_out'],
                front_tx['amount_out'],
                target_tx['gasPrice'] - 1
            )
            
            return [front_tx, back_tx]
```

## 📊 On-Chain Analytics

### DeFi Analytics Dashboard
```python
class DeFiAnalytics:
    def __init__(self, web3_provider):
        self.w3 = Web3(Web3.HTTPProvider(web3_provider))
        self.graph_endpoints = {
            'uniswap': "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3",
            'sushiswap': "https://api.thegraph.com/subgraphs/name/sushiswap/exchange",
            'aave': "https://api.thegraph.com/subgraphs/name/aave/protocol-v2"
        }
    
    async def analyze_defi_trends(self, time_period='7d'):
        """
        Phân tích DeFi trends
        """
        analytics = {}
        
        # TVL trends
        tvl_data = await self.get_tvl_trends(time_period)
        analytics['tvl_trends'] = tvl_data
        
        # Volume trends
        volume_data = await self.get_volume_trends(time_period)
        analytics['volume_trends'] = volume_data
        
        # Yield trends
        yield_data = await self.get_yield_trends(time_period)
        analytics['yield_trends'] = yield_data
        
        # Top protocols
        top_protocols = await self.get_top_protocols_by_tvl()
        analytics['top_protocols'] = top_protocols
        
        return analytics
    
    async def get_tvl_trends(self, time_period):
        """
        Lấy TVL trends từ The Graph
        """
        query = """
        {
          protocols(first: 100, orderBy: totalValueLockedUSD, orderDirection: desc) {
            id
            name
            totalValueLockedUSD
            totalVolumeUSD
          }
        }
        """
        
        tvl_data = {}
        for protocol, endpoint in self.graph_endpoints.items():
            response = await self.query_graph(endpoint, query)
            tvl_data[protocol] = response['data']['protocols']
        
        return tvl_data
    
    async def track_whale_movements(self, min_amount=1000000):
        """
        Track whale movements trong DeFi
        """
        whale_movements = []
        
        # Monitor large transactions
        latest_block = self.w3.eth.block_number
        
        for block_num in range(latest_block - 100, latest_block):
            block = self.w3.eth.get_block(block_num, full_transactions=True)
            
            for tx in block.transactions:
                if tx.value > self.w3.toWei(min_amount, 'ether'):
                    # Analyze transaction
                    analysis = await self.analyze_large_transaction(tx)
                    
                    if analysis['is_defi']:
                        whale_movements.append({
                            'tx_hash': tx.hash.hex(),
                            'from': tx['from'],
                            'to': tx['to'],
                            'value': tx.value,
                            'protocol': analysis['protocol'],
                            'action': analysis['action']
                        })
        
        return whale_movements
    
    async def predict_token_price_impact(self, token, trade_size):
        """
        Predict price impact của large trade
        """
        # Lấy liquidity data
        liquidity_data = await self.get_token_liquidity(token)
        
        # Calculate price impact
        price_impact = self.calculate_price_impact(
            trade_size, liquidity_data['total_liquidity']
        )
        
        return {
            'token': token,
            'trade_size': trade_size,
            'predicted_price_impact': price_impact,
            'liquidity_sources': liquidity_data['sources']
        }
```

## 🔐 Security và Risk Management

### Smart Contract Risk Assessment
```python
class DeFiRiskAssessment:
    def __init__(self):
        self.risk_factors = {
            'smart_contract_risk': 0.3,
            'liquidity_risk': 0.2,
            'regulatory_risk': 0.2,
            'oracle_risk': 0.15,
            'governance_risk': 0.15
        }
    
    async def assess_protocol_risk(self, protocol_address):
        """
        Đánh giá risk của DeFi protocol
        """
        risk_scores = {}
        
        # Smart contract risk
        sc_risk = await self.assess_smart_contract_risk(protocol_address)
        risk_scores['smart_contract'] = sc_risk
        
        # Liquidity risk
        liquidity_risk = await self.assess_liquidity_risk(protocol_address)
        risk_scores['liquidity'] = liquidity_risk
        
        # Oracle risk
        oracle_risk = await self.assess_oracle_risk(protocol_address)
        risk_scores['oracle'] = oracle_risk
        
        # Governance risk
        governance_risk = await self.assess_governance_risk(protocol_address)
        risk_scores['governance'] = governance_risk
        
        # Composite risk score
        composite_risk = sum(
            risk_scores[factor] * self.risk_factors[factor.replace('_', '_')]
            for factor in risk_scores
        )
        
        return {
            'protocol_address': protocol_address,
            'risk_scores': risk_scores,
            'composite_risk': composite_risk,
            'risk_rating': self.get_risk_rating(composite_risk)
        }
    
    async def assess_smart_contract_risk(self, protocol_address):
        """
        Đánh giá smart contract risk
        """
        # Kiểm tra audit history
        audit_score = await self.check_audit_history(protocol_address)
        
        # Kiểm tra code complexity
        complexity_score = await self.analyze_code_complexity(protocol_address)
        
        # Kiểm tra upgrade pattern
        upgrade_risk = await self.assess_upgrade_risk(protocol_address)
        
        # Kiểm tra time locks
        timelock_score = await self.check_timelock_protection(protocol_address)
        
        return {
            'audit_score': audit_score,
            'complexity_score': complexity_score,
            'upgrade_risk': upgrade_risk,
            'timelock_score': timelock_score,
            'overall_score': np.mean([audit_score, complexity_score, upgrade_risk, timelock_score])
        }
```

## 🔮 Future of DeFi

### Emerging Trends
1. **Layer 2 Solutions**: Polygon, Arbitrum, Optimism
2. **Cross-chain DeFi**: Interoperability protocols
3. **Institutional DeFi**: Regulated DeFi products
4. **Real-world Assets**: Tokenization of traditional assets
5. **Decentralized Derivatives**: Complex financial instruments

### Next-Generation Protocols
- **Automated Market Making 2.0**: Dynamic fee models
- **Decentralized Options**: On-chain derivatives
- **Synthetic Assets**: Exposure to any asset class
- **Yield Tokenization**: Separate yield from principal
- **DAO Treasury Management**: Decentralized asset management

---

**Tags:** #defi #blockchain #cryptocurrency #yield-farming #arbitrage #mev
**Ngày tạo:** 2024-12-19  
**Trạng thái:** #cutting-edge