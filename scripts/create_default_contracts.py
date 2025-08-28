#!/usr/bin/env python3
"""
Script to create default contracts for all MeRNSTA agents.
This ensures that all agents have properly defined contracts for Phase 26.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.agent_contract import create_default_contracts, AgentContract

def main():
    """Generate default contracts for all known agents."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Ensure contracts directory exists
    contracts_dir = project_root / "output" / "contracts"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating default contracts in: {contracts_dir}")
    
    # Create default contracts
    default_contracts = create_default_contracts()
    
    created_count = 0
    updated_count = 0
    
    for agent_name, contract in default_contracts.items():
        contract_file = contracts_dir / f"{agent_name}.json"
        
        if contract_file.exists():
            print(f"⚠️  Contract already exists for {agent_name}, updating...")
            updated_count += 1
        else:
            print(f"✅ Creating new contract for {agent_name}")
            created_count += 1
        
        # Save the contract
        contract.save_to_file(contract_file)
        
        # Print contract summary
        summary = contract.get_summary()
        print(f"   Purpose: {summary['purpose']}")
        print(f"   Top capabilities: {[cap[0] for cap in summary['top_capabilities']]}")
        print()
    
    print(f"🎉 Contract creation complete!")
    print(f"   Created: {created_count} new contracts")
    print(f"   Updated: {updated_count} existing contracts")
    print(f"   Total: {len(default_contracts)} contracts")
    
    # List all contract files
    contract_files = list(contracts_dir.glob("*.json"))
    print(f"\n📁 Contract files in {contracts_dir}:")
    for contract_file in sorted(contract_files):
        print(f"   - {contract_file.name}")

if __name__ == "__main__":
    main()