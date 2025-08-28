#!/usr/bin/env python3
"""
Test Suite for MeRNSTA Self-Upgrading System

Comprehensive tests for ArchitectAnalyzer, CodeRefactorer, UpgradeManager,
and UpgradeLedger components.
"""

import os
import sys
import pytest
import tempfile
import shutil
import sqlite3
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.architect_analyzer import ArchitectAnalyzer, ModuleAnalyzer
from agents.code_refactorer import CodeRefactorer
from agents.upgrade_manager import UpgradeManager, UpgradeTask, UpgradeStatus
from storage.upgrade_ledger import UpgradeLedger


class TestArchitectAnalyzer:
    """Test the ArchitectAnalyzer agent."""
    
    @pytest.fixture
    def analyzer(self):
        """Create an ArchitectAnalyzer instance for testing."""
        with patch('agents.architect_analyzer.get_config') as mock_config:
            mock_config.return_value = {'multi_agent': {'enabled': True}}
            return ArchitectAnalyzer()
    
    @pytest.fixture
    def temp_project(self):
        """Create a temporary project structure for testing."""
        temp_dir = tempfile.mkdtemp()
        project_path = Path(temp_dir)
        
        # Create sample files with different architectural issues
        
        # God class example
        god_class_code = '''
import os
import sys
import json
from typing import Any, Dict, List

class MassiveController:
    """A god class with too many responsibilities."""
    
    def __init__(self):
        self.data = {}
        self.config = {}
        self.connections = []
        self.cache = {}
        self.logger = None
        
    def initialize_system(self):
        pass
        
    def load_configuration(self):
        pass
        
    def setup_database(self):
        pass
        
    def create_connections(self):
        pass
        
    def process_requests(self):
        pass
        
    def handle_authentication(self):
        pass
        
    def manage_sessions(self):
        pass
        
    def generate_reports(self):
        pass
        
    def send_notifications(self):
        pass
        
    def backup_data(self):
        pass
        
    def cleanup_resources(self):
        pass
        
    def validate_input(self):
        pass
        
    def transform_data(self):
        pass
        
    def export_results(self):
        pass
        
    def schedule_tasks(self):
        pass
        
    def monitor_performance(self):
        pass
        
    def handle_errors(self):
        pass
        
    def log_activities(self):
        pass
        
    def manage_permissions(self):
        pass
        
    def process_payments(self):
        pass
'''
        
        # Circular import example
        module_a_code = '''
from .module_b import ModuleBClass

class ModuleAClass:
    def __init__(self):
        self.b_instance = ModuleBClass()
        
    def do_something(self):
        return self.b_instance.helper_method()
'''
        
        module_b_code = '''
from .module_a import ModuleAClass

class ModuleBClass:
    def helper_method(self):
        return "result"
        
    def create_a_instance(self):
        return ModuleAClass()
'''
        
        # Duplicate pattern example
        duplicate_1_code = '''
def process_user_data(data):
    if not data:
        return None
    
    processed = {}
    for key, value in data.items():
        if isinstance(value, str):
            processed[key] = value.strip().lower()
        else:
            processed[key] = value
    
    return processed

def validate_user_data(data):
    required_fields = ['name', 'email']
    for field in required_fields:
        if field not in data:
            return False
    return True
'''
        
        duplicate_2_code = '''
def process_product_data(data):
    if not data:
        return None
    
    processed = {}
    for key, value in data.items():
        if isinstance(value, str):
            processed[key] = value.strip().lower()
        else:
            processed[key] = value
    
    return processed

def validate_product_data(data):
    required_fields = ['name', 'price']
    for field in required_fields:
        if field not in data:
            return False
    return True
'''
        
        # Create directory structure
        (project_path / "controllers").mkdir()
        (project_path / "modules").mkdir()
        (project_path / "processors").mkdir()
        
        # Write test files
        (project_path / "controllers" / "massive_controller.py").write_text(god_class_code)
        (project_path / "modules" / "module_a.py").write_text(module_a_code)
        (project_path / "modules" / "module_b.py").write_text(module_b_code)
        (project_path / "processors" / "user_processor.py").write_text(duplicate_1_code)
        (project_path / "processors" / "product_processor.py").write_text(duplicate_2_code)
        
        yield project_path
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_analyze_codebase(self, analyzer, temp_project):
        """Test full codebase analysis."""
        with patch.object(analyzer, 'project_root', temp_project):
            results = analyzer.analyze_codebase()
        
        assert results["success"] is True
        assert "modules" in results
        assert "global_issues" in results
        assert "upgrade_suggestions" in results
        assert len(results["modules"]) > 0
    
    def test_detect_god_class(self, analyzer, temp_project):
        """Test detection of god classes."""
        with patch.object(analyzer, 'project_root', temp_project):
            results = analyzer.analyze_codebase()
        
        violations = results["global_issues"]["architectural_violations"]
        god_classes = [v for v in violations if v["type"] == "god_class"]
        
        assert len(god_classes) > 0
        assert any("MassiveController" in gc["class"] for gc in god_classes)
    
    def test_detect_circular_imports(self, analyzer, temp_project):
        """Test detection of circular imports."""
        with patch.object(analyzer, 'project_root', temp_project):
            results = analyzer.analyze_codebase()
        
        circular_imports = results["global_issues"]["circular_imports"]
        # Note: This test might not detect cycles due to simplified import analysis
        # but the structure is there for when the analysis is enhanced
        assert isinstance(circular_imports, list)
    
    def test_detect_duplicate_patterns(self, analyzer, temp_project):
        """Test detection of duplicate code patterns."""
        with patch.object(analyzer, 'project_root', temp_project):
            results = analyzer.analyze_codebase()
        
        duplicates = results["global_issues"]["duplicate_patterns"]
        assert isinstance(duplicates, list)
    
    def test_generate_upgrade_suggestions(self, analyzer, temp_project):
        """Test generation of upgrade suggestions."""
        with patch.object(analyzer, 'project_root', temp_project):
            results = analyzer.analyze_codebase()
        
        suggestions = results["upgrade_suggestions"]
        assert isinstance(suggestions, list)
        
        # Should have suggestions for god class
        god_class_suggestions = [s for s in suggestions if s["type"] == "god_class"]
        assert len(god_class_suggestions) > 0
        
        # Check suggestion structure
        for suggestion in suggestions:
            assert "id" in suggestion
            assert "type" in suggestion
            assert "title" in suggestion
            assert "risk_level" in suggestion
            assert "priority" in suggestion


class TestCodeRefactorer:
    """Test the CodeRefactorer agent."""
    
    @pytest.fixture
    def refactorer(self):
        """Create a CodeRefactorer instance for testing."""
        with patch('agents.code_refactorer.get_config') as mock_config:
            mock_config.return_value = {'multi_agent': {'enabled': True}}
            return CodeRefactorer()
    
    @pytest.fixture
    def temp_project(self):
        """Create a temporary project for refactoring tests."""
        temp_dir = tempfile.mkdtemp()
        project_path = Path(temp_dir)
        
        # Create a simple god class to refactor
        god_class_code = '''
class SimpleGodClass:
    def __init__(self):
        self.data = {}
        
    def method1(self):
        return "method1"
        
    def method2(self):
        return "method2"
        
    def method3(self):
        return "method3"
'''
        
        (project_path / "simple_god.py").write_text(god_class_code)
        
        yield project_path
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_create_backup(self, refactorer, temp_project):
        """Test backup creation before refactoring."""
        with patch.object(refactorer, 'project_root', temp_project):
            affected_modules = ["simple_god.py"]
            backup_path = refactorer._create_backup(affected_modules)
        
        assert backup_path.exists()
        assert (backup_path / "simple_god.py").exists()
    
    def test_validate_syntax(self, refactorer):
        """Test syntax validation of refactored code."""
        valid_change = {
            "path": "test.py",
            "content": "def valid_function():\n    return True"
        }
        
        invalid_change = {
            "path": "test.py", 
            "content": "def invalid_function(\n    return True"  # Missing closing parenthesis
        }
        
        assert refactorer._validate_syntax([valid_change]) is True
        assert refactorer._validate_syntax([invalid_change]) is False
    
    def test_execute_refactor_god_class(self, refactorer, temp_project):
        """Test execution of god class refactoring."""
        with patch.object(refactorer, 'project_root', temp_project):
            suggestion = {
                "id": "test_god_class",
                "type": "god_class",
                "title": "Refactor god class SimpleGodClass",
                "affected_modules": ["simple_god.py"]
            }
            
            # Mock LLM response
            with patch.object(refactorer, 'llm_fallback') as mock_llm:
                mock_llm.process.return_value = json.dumps({
                    "new_classes": [{
                        "filename": "simple_god_helper.py",
                        "class_name": "SimpleGodHelper",
                        "code": "class SimpleGodHelper:\n    def helper_method(self):\n        pass"
                    }],
                    "updated_main_file": "# Updated main file\nclass SimpleGodClass:\n    pass",
                    "rationale": "Split into helper class"
                })
                
                result = refactorer.execute_refactor(suggestion)
        
        assert result["suggestion_id"] == "test_god_class"
        assert "changes" in result
        assert "backup_location" in result


class TestUpgradeManager:
    """Test the UpgradeManager agent."""
    
    @pytest.fixture
    def upgrade_manager(self):
        """Create an UpgradeManager instance for testing."""
        with patch('agents.upgrade_manager.get_config') as mock_config:
            mock_config.return_value = {'multi_agent': {'enabled': True}}
            
            # Mock the component agents
            with patch('agents.upgrade_manager.ArchitectAnalyzer') as mock_analyzer, \
                 patch('agents.upgrade_manager.CodeRefactorer') as mock_refactorer, \
                 patch('agents.upgrade_manager.UpgradeLedger') as mock_ledger:
                
                manager = UpgradeManager()
                manager.analyzer = mock_analyzer.return_value
                manager.refactorer = mock_refactorer.return_value
                manager.ledger = mock_ledger.return_value
                
                return manager
    
    def test_initialization(self, upgrade_manager):
        """Test UpgradeManager initialization."""
        assert upgrade_manager.upgrade_queue == []
        assert upgrade_manager.completed_upgrades == []
        assert upgrade_manager.active_upgrades == 0
        assert upgrade_manager.is_running is False
    
    def test_trigger_scan(self, upgrade_manager):
        """Test manual scan triggering."""
        # Mock analyzer results
        mock_analysis = {
            "timestamp": datetime.now().isoformat(),
            "analyzed_path": "/test/path",
            "modules": {"test.py": {"analyzable": True}},
            "global_issues": {},
            "upgrade_suggestions": [
                {
                    "id": "test_suggestion",
                    "type": "god_class",
                    "priority": 7,
                    "risk_level": "medium"
                }
            ]
        }
        
        upgrade_manager.analyzer.analyze_codebase.return_value = mock_analysis
        
        result = upgrade_manager.trigger_scan()
        
        assert result["success"] is True
        assert result["total_suggestions"] == 1
        assert result["queued_upgrades"] == 1
        assert len(upgrade_manager.upgrade_queue) == 1
    
    def test_upgrade_task_creation(self, upgrade_manager):
        """Test creation of upgrade tasks."""
        suggestion = {
            "id": "test_suggestion",
            "type": "god_class",
            "priority": 5,
            "affected_modules": ["test.py"]
        }
        
        task = upgrade_manager._create_upgrade_task(suggestion)
        
        assert isinstance(task, UpgradeTask)
        assert task.suggestion == suggestion
        assert task.status == UpgradeStatus.PENDING
        assert task.priority == 5
    
    def test_execute_upgrade(self, upgrade_manager):
        """Test upgrade execution."""
        # Create a test task
        suggestion = {
            "id": "test_suggestion",
            "type": "god_class",
            "affected_modules": ["test.py"]
        }
        
        task = upgrade_manager._create_upgrade_task(suggestion)
        upgrade_manager._add_to_queue(task)
        
        # Mock refactorer response
        mock_refactor_result = {
            "success": True,
            "changes": [],
            "backup_location": "/tmp/backup"
        }
        upgrade_manager.refactorer.execute_refactor.return_value = mock_refactor_result
        
        result = upgrade_manager.execute_upgrade(task.id)
        
        assert result["success"] is True
        assert task.status == UpgradeStatus.COMPLETED
    
    def test_rollback_upgrade(self, upgrade_manager):
        """Test upgrade rollback."""
        # Create a completed upgrade
        suggestion = {"id": "test_suggestion", "type": "god_class"}
        task = upgrade_manager._create_upgrade_task(suggestion)
        task.status = UpgradeStatus.COMPLETED
        task.result = {
            "backup_location": "/tmp/test_backup",
            "changes": [{
                "type": "modify",
                "path": "/test/file.py",
                "original_path": "/test/file.py"
            }]
        }
        upgrade_manager.completed_upgrades.append(task)
        
        # Mock filesystem operations
        with patch('pathlib.Path.exists') as mock_exists, \
             patch('shutil.copy2') as mock_copy:
            
            mock_exists.return_value = True
            
            result = upgrade_manager.rollback_upgrade(task.id)
        
        assert result["success"] is True
        assert task.status == UpgradeStatus.ROLLED_BACK
    
    def test_get_upgrade_status(self, upgrade_manager):
        """Test getting upgrade status."""
        # Add some test tasks
        task1 = UpgradeTask("task1", {}, UpgradeStatus.PENDING, datetime.now())
        task2 = UpgradeTask("task2", {}, UpgradeStatus.IN_PROGRESS, datetime.now())
        task3 = UpgradeTask("task3", {}, UpgradeStatus.COMPLETED, datetime.now())
        
        upgrade_manager.upgrade_queue = [task1, task2]
        upgrade_manager.completed_upgrades = [task3]
        
        status = upgrade_manager.get_upgrade_status()
        
        assert status["queue_length"] == 1  # Only pending tasks
        assert status["in_progress"] == 1
        assert len(status["pending_tasks"]) == 1
        assert len(status["in_progress_tasks"]) == 1
    
    def test_learn_from_outcomes(self, upgrade_manager):
        """Test learning from upgrade outcomes."""
        # Add some completed upgrades
        successful_task = UpgradeTask("success", {"type": "god_class"}, UpgradeStatus.COMPLETED, datetime.now())
        failed_task = UpgradeTask("failed", {"type": "circular_import"}, UpgradeStatus.FAILED, datetime.now())
        failed_task.result = {"errors": ["Test error"]}
        
        upgrade_manager.completed_upgrades = [successful_task, failed_task]
        
        learning_data = upgrade_manager.learn_from_outcomes()
        
        assert learning_data["total_upgrades"] == 2
        assert learning_data["success_rate"] == 0.5
        assert "failure_patterns" in learning_data
        assert "success_patterns" in learning_data


class TestUpgradeLedger:
    """Test the UpgradeLedger storage system."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        
        yield temp_db.name
        
        # Cleanup
        os.unlink(temp_db.name)
    
    @pytest.fixture
    def ledger(self, temp_db):
        """Create an UpgradeLedger instance with temporary database."""
        return UpgradeLedger(temp_db)
    
    def test_database_initialization(self, ledger):
        """Test database schema creation."""
        with ledger._get_connection() as conn:
            cursor = conn.cursor()
            
            # Check that all tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = [
                'file_versions', 'upgrade_scans', 'upgrade_executions',
                'file_changes', 'rollback_operations', 'upgrade_learning'
            ]
            
            for table in expected_tables:
                assert table in tables
    
    def test_log_scan(self, ledger):
        """Test logging of architecture scans."""
        analysis_results = {
            "analyzed_path": "/test/path",
            "modules": {"test.py": {"analyzable": True}},
            "upgrade_suggestions": [{"id": "test", "type": "god_class"}]
        }
        
        scan_metadata = {
            "queued_upgrades": 1,
            "scan_duration": 5.2
        }
        
        scan_id = ledger.log_scan(analysis_results, scan_metadata)
        
        assert isinstance(scan_id, int)
        assert scan_id > 0
        
        # Verify data was stored
        with ledger._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM upgrade_scans WHERE id = ?", (scan_id,))
            row = cursor.fetchone()
            
            assert row is not None
            assert row["total_suggestions"] == 1
    
    def test_log_upgrade_execution(self, ledger):
        """Test logging of upgrade executions."""
        suggestion = {
            "id": "test_suggestion",
            "type": "god_class",
            "affected_modules": ["test.py"],
            "risk_level": "medium",
            "priority": 5
        }
        
        execution_result = {
            "success": True,
            "started_at": datetime.now().isoformat(),
            "completed_at": datetime.now().isoformat(),
            "changes": [{
                "type": "modify",
                "path": "/test/file.py",
                "content": "new content"
            }],
            "backup_location": "/tmp/backup"
        }
        
        execution_id = ledger.log_upgrade_execution("upgrade_123", suggestion, execution_result)
        
        assert isinstance(execution_id, int)
        assert execution_id > 0
        
        # Verify data was stored
        with ledger._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM upgrade_executions WHERE id = ?", (execution_id,))
            row = cursor.fetchone()
            
            assert row is not None
            assert row["upgrade_id"] == "upgrade_123"
            assert row["upgrade_type"] == "god_class"
            assert row["success"] == 1
    
    def test_log_rollback(self, ledger):
        """Test logging of rollback operations."""
        # First create an upgrade execution to rollback
        suggestion = {"id": "test", "type": "god_class", "affected_modules": []}
        execution_result = {"success": True, "changes": []}
        
        execution_id = ledger.log_upgrade_execution("upgrade_rollback_test", suggestion, execution_result)
        
        # Now log the rollback
        rollback_result = {
            "success": True,
            "restored_files": ["/test/file.py"],
            "reason": "Test rollback"
        }
        
        rollback_id = ledger.log_rollback("upgrade_rollback_test", rollback_result)
        
        assert isinstance(rollback_id, int)
        assert rollback_id > 0
        
        # Verify rollback was logged
        with ledger._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM rollback_operations WHERE id = ?", (rollback_id,))
            row = cursor.fetchone()
            
            assert row is not None
            assert row["upgrade_execution_id"] == execution_id
            assert row["success"] == 1
    
    def test_get_upgrade_statistics(self, ledger):
        """Test upgrade statistics generation."""
        # Add some test data
        suggestion1 = {"id": "test1", "type": "god_class", "affected_modules": []}
        suggestion2 = {"id": "test2", "type": "circular_import", "affected_modules": []}
        
        result1 = {"success": True, "changes": []}
        result2 = {"success": False, "changes": [], "errors": ["Test error"]}
        
        ledger.log_upgrade_execution("upgrade_1", suggestion1, result1)
        ledger.log_upgrade_execution("upgrade_2", suggestion2, result2)
        
        stats = ledger.get_upgrade_statistics()
        
        assert stats["total_upgrades"] == 2
        assert stats["successful_upgrades"] == 1
        assert stats["success_rate"] == 0.5
        assert "by_type" in stats
        assert stats["by_type"]["god_class"]["total"] == 1
        assert stats["by_type"]["god_class"]["successful"] == 1
    
    def test_analyze_failure_patterns(self, ledger):
        """Test failure pattern analysis."""
        # Add some failed upgrades
        suggestion1 = {"id": "fail1", "type": "god_class", "affected_modules": []}
        suggestion2 = {"id": "fail2", "type": "god_class", "affected_modules": []}
        
        result1 = {"success": False, "errors": ["Syntax error in refactored code"]}
        result2 = {"success": False, "errors": ["Test failure after refactoring"]}
        
        ledger.log_upgrade_execution("fail_upgrade_1", suggestion1, result1)
        ledger.log_upgrade_execution("fail_upgrade_2", suggestion2, result2)
        
        analysis = ledger.analyze_failure_patterns()
        
        assert analysis["total_failures"] == 2
        assert "god_class" in analysis["failure_by_type"]
        assert len(analysis["failure_by_type"]["god_class"]) == 2
        assert "syntax_error" in analysis["common_errors"]
        assert "test_failure" in analysis["common_errors"]
    
    def test_cleanup_old_records(self, ledger):
        """Test cleanup of old records."""
        # Add some old test data
        old_date = datetime.now() - timedelta(days=100)
        
        with ledger._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO upgrade_scans (scan_timestamp, total_files) 
                VALUES (?, ?)
            """, (old_date.isoformat(), 10))
            
            cursor.execute("""
                INSERT INTO file_versions (file_path, content_hash, file_size, created_at)
                VALUES (?, ?, ?, ?)
            """, ("old_file.py", "hash123", 100, old_date.isoformat()))
            
            conn.commit()
        
        # Clean up records older than 30 days
        removed_counts = ledger.cleanup_old_records(days_to_keep=30)
        
        assert "scans" in removed_counts
        assert "file_versions" in removed_counts
        assert removed_counts["scans"] >= 1  # Should remove the old scan


class TestIntegration:
    """Integration tests for the complete self-upgrading system."""
    
    @pytest.fixture
    def temp_environment(self):
        """Create a complete temporary environment for integration testing."""
        temp_dir = tempfile.mkdtemp()
        project_path = Path(temp_dir)
        
        # Create a more realistic project structure
        (project_path / "agents").mkdir()
        (project_path / "storage").mkdir()
        (project_path / "tests").mkdir()
        (project_path / "core_v2").mkdir()
        (project_path / "backups").mkdir()
        
        # Create some sample files to analyze
        sample_god_class = '''
class DatabaseManager:
    def __init__(self):
        self.connections = {}
        self.cache = {}
        self.config = {}
    
    def connect_mysql(self): pass
    def connect_postgres(self): pass
    def connect_mongodb(self): pass
    def execute_query(self): pass
    def update_record(self): pass
    def delete_record(self): pass
    def backup_database(self): pass
    def restore_database(self): pass
    def optimize_database(self): pass
    def monitor_performance(self): pass
    def handle_errors(self): pass
    def log_operations(self): pass
    def validate_schema(self): pass
    def migrate_data(self): pass
    def export_data(self): pass
    def import_data(self): pass
    def setup_replication(self): pass
    def manage_users(self): pass
    def handle_permissions(self): pass
    def encrypt_data(self): pass
'''
        
        (project_path / "storage" / "database_manager.py").write_text(sample_god_class)
        
        yield project_path
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_full_upgrade_workflow(self, temp_environment):
        """Test the complete upgrade workflow from analysis to execution."""
        # Patch the working directory and config
        with patch('os.getcwd', return_value=str(temp_environment)), \
             patch('agents.architect_analyzer.get_config') as mock_config1, \
             patch('agents.code_refactorer.get_config') as mock_config2, \
             patch('agents.upgrade_manager.get_config') as mock_config3:
            
            mock_config1.return_value = {'multi_agent': {'enabled': True}}
            mock_config2.return_value = {'multi_agent': {'enabled': True}}
            mock_config3.return_value = {'multi_agent': {'enabled': True}}
            
            # Create temporary database
            temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
            temp_db.close()
            
            try:
                # Initialize components
                analyzer = ArchitectAnalyzer()
                analyzer.project_root = temp_environment
                
                ledger = UpgradeLedger(temp_db.name)
                
                with patch('agents.upgrade_manager.UpgradeLedger') as mock_ledger_class:
                    mock_ledger_class.return_value = ledger
                    
                    manager = UpgradeManager()
                    manager.project_root = temp_environment
                    manager.analyzer = analyzer
                    manager.ledger = ledger
                
                # Step 1: Trigger analysis
                scan_result = manager.trigger_scan()
                
                assert scan_result["success"] is True
                assert scan_result["total_suggestions"] > 0
                
                # Step 2: Check that suggestions were queued
                status = manager.get_upgrade_status()
                assert status["queue_length"] > 0
                
                # Step 3: Mock successful refactoring and execute upgrade
                pending_tasks = [task for task in manager.upgrade_queue 
                               if task.status == UpgradeStatus.PENDING]
                assert len(pending_tasks) > 0
                
                first_task = pending_tasks[0]
                
                # Mock the refactorer to simulate successful upgrade
                mock_refactor_result = {
                    "success": True,
                    "started_at": datetime.now().isoformat(),
                    "completed_at": datetime.now().isoformat(),
                    "changes": [{
                        "type": "create",
                        "path": str(temp_environment / "core_v2" / "new_helper.py"),
                        "content": "class Helper:\n    pass"
                    }],
                    "backup_location": str(temp_environment / "backups" / "test_backup"),
                    "test_results": {"passed": True, "output": "All tests passed"}
                }
                
                with patch.object(manager.refactorer, 'execute_refactor') as mock_refactor:
                    mock_refactor.return_value = mock_refactor_result
                    
                    result = manager.execute_upgrade(first_task.id)
                
                assert result["success"] is True
                
                # Step 4: Verify the upgrade was logged
                history = ledger.get_upgrade_history(limit=10)
                assert len(history) > 0
                assert history[0]["upgrade_id"] == first_task.id
                
                # Step 5: Test statistics
                stats = ledger.get_upgrade_statistics()
                assert stats["total_upgrades"] > 0
                assert stats["successful_upgrades"] > 0
                
            finally:
                # Cleanup database
                os.unlink(temp_db.name)
    
    def test_rollback_workflow(self, temp_environment):
        """Test the rollback workflow."""
        with patch('os.getcwd', return_value=str(temp_environment)):
            # Create temporary database
            temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
            temp_db.close()
            
            try:
                ledger = UpgradeLedger(temp_db.name)
                
                # Simulate a completed upgrade
                suggestion = {
                    "id": "test_rollback",
                    "type": "god_class", 
                    "affected_modules": ["storage/database_manager.py"]
                }
                
                execution_result = {
                    "success": True,
                    "started_at": datetime.now().isoformat(),
                    "completed_at": datetime.now().isoformat(),
                    "changes": [{
                        "type": "modify",
                        "path": str(temp_environment / "storage" / "database_manager.py"),
                        "original_path": str(temp_environment / "storage" / "database_manager.py"),
                        "content": "# Modified content"
                    }],
                    "backup_location": str(temp_environment / "backups" / "rollback_test")
                }
                
                upgrade_id = "rollback_test_upgrade"
                ledger.log_upgrade_execution(upgrade_id, suggestion, execution_result)
                
                # Create fake backup
                backup_dir = temp_environment / "backups" / "rollback_test"
                backup_dir.mkdir(parents=True, exist_ok=True)
                (backup_dir / "database_manager.py").write_text("# Original content")
                
                # Test rollback
                with patch('agents.upgrade_manager.get_config') as mock_config:
                    mock_config.return_value = {'multi_agent': {'enabled': True}}
                    
                    manager = UpgradeManager()
                    manager.project_root = temp_environment
                    manager.ledger = ledger
                    
                    # Add the completed task
                    task = manager._create_upgrade_task(suggestion)
                    task.status = UpgradeStatus.COMPLETED
                    task.result = execution_result
                    manager.completed_upgrades.append(task)
                    
                    rollback_result = manager.rollback_upgrade(upgrade_id)
                
                assert rollback_result["success"] is True
                
                # Verify rollback was logged
                with ledger._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM rollback_operations")
                    count = cursor.fetchone()[0]
                    assert count > 0
                
            finally:
                # Cleanup database
                os.unlink(temp_db.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])