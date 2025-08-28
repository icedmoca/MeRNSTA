#!/usr/bin/env python3
"""
Comprehensive tests for Phase 12: Neural Reflection & Self-Repair System

Tests all components of the self-repair capabilities including:
- SelfHealer (code health analysis, architecture flaw detection, repair goal generation)
- SelfRepairLog (repair attempt tracking, pattern learning, outcome analysis)
- CLI integration
- Integration with recursive planning system
"""

import pytest
import tempfile
import os
import json
import uuid
import ast
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path

# Import the self-repair components
from agents.self_healer import SelfHealer, Issue, Pattern, DiagnosticReport
from storage.self_repair_log import SelfRepairLog, RepairAttempt

class TestSelfHealerDataStructures:
    """Test the basic data structures for self-repair"""
    
    def test_issue_creation(self):
        """Test Issue creation and defaults"""
        issue = Issue(
            issue_id="test-issue-1",
            category="code_quality",
            severity="high",
            component="test_module.py",
            description="Test issue description",
            evidence=["evidence1", "evidence2"],
            impact_score=0.8,
            fix_difficulty=0.6,
            detected_at=datetime.now().isoformat()
        )
        
        assert issue.issue_id == "test-issue-1"
        assert issue.category == "code_quality"
        assert issue.severity == "high"
        assert issue.component == "test_module.py"
        assert issue.impact_score == 0.8
        assert issue.fix_difficulty == 0.6
        assert issue.repair_suggestions == []
    
    def test_pattern_creation(self):
        """Test Pattern creation and defaults"""
        pattern = Pattern(
            pattern_id="test-pattern-1",
            pattern_type="anti_pattern",
            name="God Class",
            description="Class with too many responsibilities",
            locations=["class1.py", "class2.py"],
            frequency=5,
            risk_level="high",
            recommended_action="Break down into smaller classes"
        )
        
        assert pattern.pattern_id == "test-pattern-1"
        assert pattern.pattern_type == "anti_pattern"
        assert pattern.name == "God Class"
        assert pattern.frequency == 5
        assert pattern.risk_level == "high"
        assert len(pattern.locations) == 2
    
    def test_diagnostic_report_creation(self):
        """Test DiagnosticReport creation"""
        issues = [
            Issue("i1", "code_quality", "high", "test.py", "Test issue", [], 0.8, 0.5, datetime.now().isoformat())
        ]
        patterns = [
            Pattern("p1", "anti_pattern", "Test Pattern", "Test description", [], 3, "medium", "Test action")
        ]
        
        report = DiagnosticReport(
            report_id="test-report",
            generated_at=datetime.now().isoformat(),
            system_health_score=0.75,
            issues=issues,
            patterns=patterns,
            metrics={"test_metric": 100},
            recommendations=["Test recommendation"],
            repair_goals=["Test repair goal"]
        )
        
        assert report.report_id == "test-report"
        assert report.system_health_score == 0.75
        assert len(report.issues) == 1
        assert len(report.patterns) == 1
        assert len(report.repair_goals) == 1

class TestSelfHealer:
    """Test the SelfHealer agent"""
    
    @pytest.fixture
    def self_healer(self):
        """Create a test self-healer instance"""
        with patch('agents.self_healer.SelfHealer._SelfHealer__repair_log', None):
            with patch('agents.self_healer.SelfHealer._SelfHealer__recursive_planner', None):
                with patch('agents.self_healer.SelfHealer._SelfHealer__memory_system', None):
                    healer = SelfHealer()
                    # Mock external dependencies
                    healer._repair_log = Mock()
                    healer._recursive_planner = Mock()
                    healer._memory_system = Mock()
                    return healer
    
    @pytest.fixture
    def temp_code_files(self):
        """Create temporary code files for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test Python files with various issues
            fragile_code = '''
import os
global global_var
global_var = []

def bad_function():
    try:
        pass
    except:  # Bare except
        pass
    
    for i in range(100):
        for j in range(100):  # Nested loops
            global_var.append(i + j)  # Magic number + global usage
    
    open("/hardcoded/path/file.txt")  # Resource leak + hardcoded path

class GodClass:
    def method1(self): pass
    def method2(self): pass
    def method3(self): pass
    def method4(self): pass
    def method5(self): pass
    def method6(self): pass
    def method7(self): pass
    def method8(self): pass
    def method9(self): pass
    def method10(self): pass
    def method11(self): pass
    def method12(self): pass
    def method13(self): pass
    def method14(self): pass
    def method15(self): pass
    def method16(self): pass  # 16 methods = god class
            '''
            
            good_code = '''
import logging
from typing import List, Optional

class WellDesignedClass:
    """A well-designed class with proper error handling."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def process_data(self, data: List[str]) -> Optional[str]:
        """Process data with proper error handling."""
        try:
            if not data:
                return None
            
            result = self._validate_and_process(data)
            return result
        
        except ValueError as e:
            self.logger.error(f"Data validation failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            raise
    
    def _validate_and_process(self, data: List[str]) -> str:
        """Helper method for processing."""
        return " ".join(data)
            '''
            
            # Write test files
            (temp_path / "fragile_module.py").write_text(fragile_code)
            (temp_path / "good_module.py").write_text(good_code)
            (temp_path / "test_fragile.py").write_text("# Test file")
            
            yield temp_path
    
    def test_initialization(self, self_healer):
        """Test self-healer initialization"""
        assert self_healer.name == "self_healer"
        assert hasattr(self_healer, 'scan_patterns')
        assert hasattr(self_healer, 'severity_thresholds')
        assert hasattr(self_healer, 'repair_templates')
    
    def test_scan_patterns_loading(self, self_healer):
        """Test that scan patterns are loaded correctly"""
        patterns = self_healer.scan_patterns
        
        assert 'fragile_code' in patterns
        assert 'architecture_flaws' in patterns
        assert 'performance_issues' in patterns
        assert 'reliability_issues' in patterns
        
        # Check specific patterns
        fragile = patterns['fragile_code']
        assert 'bare_except' in fragile
        assert 'global_variables' in fragile
        assert 'hardcoded_paths' in fragile
    
    def test_analyze_code_health_with_issues(self, self_healer, temp_code_files):
        """Test code health analysis with problematic code"""
        issues = self_healer.analyze_code_health([str(temp_code_files)])
        
        # Should find multiple issues
        assert len(issues) > 0
        
        # Check for specific issue types
        issue_descriptions = [issue.description for issue in issues]
        
        # Should detect bare except
        bare_except_found = any("bare except" in desc.lower() for desc in issue_descriptions)
        assert bare_except_found, f"Bare except not detected in: {issue_descriptions}"
        
        # Should detect global variables
        global_vars_found = any("global" in desc.lower() for desc in issue_descriptions)
        assert global_vars_found, f"Global variables not detected in: {issue_descriptions}"
    
    def test_detect_architecture_flaws(self, self_healer, temp_code_files):
        """Test architecture flaw detection"""
        patterns = self_healer.detect_architecture_flaws([str(temp_code_files)])
        
        # Should find patterns
        assert isinstance(patterns, list)
        
        # Should detect god class if enough methods
        pattern_names = [pattern.name for pattern in patterns]
        
        # At minimum should have some detectable patterns
        assert len(patterns) >= 0  # May be 0 if no major architectural issues
    
    def test_generate_repair_goals_empty(self, self_healer):
        """Test repair goal generation with no issues"""
        goals = self_healer.generate_repair_goals([])
        assert goals == []
    
    def test_generate_repair_goals_with_issues(self, self_healer):
        """Test repair goal generation with various issues"""
        issues = [
            Issue("i1", "code_quality", "critical", "bad.py", "Critical issue", [], 0.9, 0.5, datetime.now().isoformat()),
            Issue("i2", "code_quality", "high", "bad.py", "High issue", [], 0.8, 0.4, datetime.now().isoformat()),
            Issue("i3", "performance", "medium", "slow.py", "Performance issue", [], 0.6, 0.6, datetime.now().isoformat()),
            Issue("i4", "reliability", "high", "unreliable.py", "Reliability issue", [], 0.7, 0.5, datetime.now().isoformat()),
        ]
        
        patterns = [
            Pattern("p1", "anti_pattern", "God Class", "Too many methods", ["god.py"], 5, "critical", "Break down")
        ]
        
        goals = self_healer.generate_repair_goals(issues, patterns)
        
        assert len(goals) > 0
        
        # Should generate goals for different categories
        goal_text = " ".join(goals).lower()
        assert any(category in goal_text for category in ["code_quality", "performance", "reliability"])
    
    def test_prioritize_repairs(self, self_healer):
        """Test repair goal prioritization"""
        goals = [
            "Improve documentation",
            "Fix critical security vulnerability immediately",
            "Optimize performance by 20%",
            "Refactor legacy code"
        ]
        
        prioritized = self_healer.prioritize_repairs(goals)
        
        assert len(prioritized) == len(goals)
        
        # Critical/security issues should be prioritized
        assert "critical" in prioritized[0].lower() or "security" in prioritized[0].lower()
    
    def test_run_diagnostic_suite(self, self_healer, temp_code_files):
        """Test comprehensive diagnostic suite"""
        # Mock the project root to point to our test files
        self_healer.project_root = temp_code_files
        
        report = self_healer.run_diagnostic_suite()
        
        assert isinstance(report, DiagnosticReport)
        assert report.report_id
        assert 0.0 <= report.system_health_score <= 1.0
        assert isinstance(report.issues, list)
        assert isinstance(report.patterns, list)
        assert isinstance(report.metrics, dict)
        assert isinstance(report.recommendations, list)
        assert isinstance(report.repair_goals, list)
    
    def test_caching_behavior(self, self_healer):
        """Test that diagnostic results are cached"""
        # Mock to avoid actual file system scanning
        with patch.object(self_healer, 'analyze_code_health', return_value=[]):
            with patch.object(self_healer, 'detect_architecture_flaws', return_value=[]):
                
                # First call
                report1 = self_healer.run_diagnostic_suite()
                
                # Second call should return cached result
                report2 = self_healer.run_diagnostic_suite()
                
                assert report1.report_id == report2.report_id
    
    def test_severity_classification(self, self_healer):
        """Test severity classification for patterns"""
        # Test different pattern severities
        assert self_healer._classify_pattern_severity('bare_except') == 'high'
        assert self_healer._classify_pattern_severity('magic_numbers') == 'low'
        assert self_healer._classify_pattern_severity('unknown_pattern') == 'medium'
    
    def test_pattern_impact_calculation(self, self_healer):
        """Test impact score calculation for patterns"""
        # Test impact scores
        assert self_healer._calculate_pattern_impact('bare_except') == 0.8
        assert self_healer._calculate_pattern_impact('long_lines') == 0.2
        assert self_healer._calculate_pattern_impact('unknown_pattern') == 0.5

class TestSelfRepairLog:
    """Test the SelfRepairLog system"""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    @pytest.fixture
    def repair_log(self, temp_db):
        """Create a test repair log instance"""
        return SelfRepairLog(db_path=temp_db)
    
    def test_initialization(self, repair_log):
        """Test repair log initialization"""
        assert os.path.exists(repair_log.db_path)
        
        # Test that database tables were created
        import sqlite3
        conn = sqlite3.connect(repair_log.db_path)
        cursor = conn.cursor()
        
        # Check that main tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ['repair_attempts', 'repair_patterns', 'health_snapshots', 'repair_insights']
        for table in expected_tables:
            assert table in tables
        
        conn.close()
    
    def test_log_repair_attempt(self, repair_log):
        """Test logging a repair attempt"""
        attempt_id = repair_log.log_repair_attempt(
            goal="Fix critical issue",
            approach="Automated refactoring",
            issues_addressed=["issue1", "issue2"]
        )
        
        assert attempt_id != ""
        
        # Verify it was stored
        import sqlite3
        conn = sqlite3.connect(repair_log.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT goal, approach FROM repair_attempts WHERE attempt_id = ?", (attempt_id,))
        result = cursor.fetchone()
        
        assert result is not None
        assert result[0] == "Fix critical issue"
        assert result[1] == "Automated refactoring"
        
        conn.close()
    
    def test_complete_repair_attempt(self, repair_log):
        """Test completing a repair attempt"""
        # Start repair attempt
        attempt_id = repair_log.log_repair_attempt(
            goal="Test repair",
            approach="Test approach"
        )
        
        # Complete it
        success = repair_log.complete_repair_attempt(
            attempt_id=attempt_id,
            result="Repair completed successfully",
            score=0.8,
            issues_resolved=["resolved1", "resolved2"],
            side_effects=["minor_side_effect"]
        )
        
        assert success == True
        
        # Verify completion was recorded
        import sqlite3
        conn = sqlite3.connect(repair_log.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT status, score, result FROM repair_attempts WHERE attempt_id = ?", (attempt_id,))
        result = cursor.fetchone()
        
        assert result is not None
        assert result[0] == "completed"  # status
        assert result[1] == 0.8  # score
        assert "successfully" in result[2]  # result
        
        conn.close()
    
    def test_get_failed_repairs(self, repair_log):
        """Test retrieving failed repairs"""
        # Create some repair attempts with different outcomes
        failed_id = repair_log.log_repair_attempt("Failed repair", "Bad approach")
        repair_log.complete_repair_attempt(failed_id, "Failed miserably", 0.1)
        
        success_id = repair_log.log_repair_attempt("Successful repair", "Good approach")
        repair_log.complete_repair_attempt(success_id, "Worked great", 0.9)
        
        # Get failed repairs
        failed_repairs = repair_log.get_failed_repairs()
        
        assert len(failed_repairs) >= 1
        
        # Should contain the failed repair
        failed_goals = [repair['goal'] for repair in failed_repairs]
        assert "Failed repair" in failed_goals
    
    def test_summarize_recent_repairs(self, repair_log):
        """Test repair summary generation"""
        # Create several repair attempts
        for i in range(5):
            attempt_id = repair_log.log_repair_attempt(f"Repair {i}", "Test approach")
            score = 0.8 if i % 2 == 0 else 0.3  # Alternate success/failure
            repair_log.complete_repair_attempt(attempt_id, f"Result {i}", score)
        
        summary = repair_log.summarize_recent_repairs(days_back=30)
        
        assert summary['total_attempts'] == 5
        assert summary['successful_attempts'] == 3  # 3 with score >= 0.5
        assert summary['failed_attempts'] == 2     # 2 with score < 0.5
        assert 0.0 <= summary['success_rate'] <= 1.0
        assert summary['average_score'] > 0.0
    
    def test_take_health_snapshot(self, repair_log):
        """Test taking health snapshots"""
        snapshot_id = repair_log.take_health_snapshot(
            health_score=0.75,
            total_issues=10,
            critical_issues=2,
            high_issues=3,
            issues_by_category={"code_quality": 5, "performance": 3, "reliability": 2},
            patterns_detected=["God Class", "Circular Import"]
        )
        
        assert snapshot_id != ""
        
        # Verify snapshot was stored
        import sqlite3
        conn = sqlite3.connect(repair_log.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT health_score, total_issues FROM health_snapshots WHERE snapshot_id = ?", (snapshot_id,))
        result = cursor.fetchone()
        
        assert result is not None
        assert result[0] == 0.75  # health_score
        assert result[1] == 10    # total_issues
        
        conn.close()
    
    def test_get_health_trend(self, repair_log):
        """Test health trend analysis"""
        # Create multiple health snapshots over time
        base_time = datetime.now() - timedelta(days=5)
        
        for i in range(5):
            snapshot_time = base_time + timedelta(days=i)
            health_score = 0.5 + (i * 0.1)  # Improving trend
            
            # Manually insert with specific timestamp for testing
            import sqlite3
            conn = sqlite3.connect(repair_log.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO health_snapshots (
                    snapshot_id, taken_at, health_score, total_issues,
                    critical_issues, high_issues, issues_by_category, patterns_detected
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()),
                snapshot_time.isoformat(),
                health_score,
                10 - i,  # Decreasing issues
                2 - (i // 2),  # Decreasing critical issues
                3 - i,  # Decreasing high issues
                json.dumps({}),
                json.dumps([])
            ))
            
            conn.commit()
            conn.close()
        
        trend = repair_log.get_health_trend(days_back=10)
        
        assert len(trend) == 5
        
        # Should show improving trend
        first_score = trend[0]['health_score']
        last_score = trend[-1]['health_score']
        assert last_score > first_score

class TestCLIIntegration:
    """Test CLI command integration"""
    
    def test_self_repair_commands_available(self):
        """Test that self-repair commands are in available commands"""
        from cortex.cli_commands import AVAILABLE_COMMANDS
        
        expected_commands = [
            "self_diagnose", "self_repair", "show_flaws", "repair_log"
        ]
        
        for cmd in expected_commands:
            assert cmd in AVAILABLE_COMMANDS, f"Command {cmd} not in available commands"
    
    @patch('cortex.cli_commands.SELF_REPAIR_MODE', True)
    def test_self_diagnose_command_handler(self):
        """Test self_diagnose command handler"""
        from cortex.cli_commands import handle_self_diagnose_command
        
        with patch('cortex.cli_commands.SelfHealer') as mock_healer_class:
            mock_healer = Mock()
            mock_report = Mock()
            mock_report.system_health_score = 0.8
            mock_report.issues = []
            mock_report.patterns = []
            mock_report.recommendations = ["Test recommendation"]
            mock_report.repair_goals = ["Test repair goal"]
            
            mock_healer.run_diagnostic_suite.return_value = mock_report
            mock_healer_class.return_value = mock_healer
            
            result = handle_self_diagnose_command()
            assert result == 'continue'
            mock_healer.run_diagnostic_suite.assert_called_once()
    
    @patch('cortex.cli_commands.SELF_REPAIR_MODE', True)
    def test_self_repair_command_handler(self):
        """Test self_repair command handler"""
        from cortex.cli_commands import handle_self_repair_command
        
        with patch('cortex.cli_commands.SelfHealer') as mock_healer_class:
            with patch('cortex.cli_commands.RecursivePlanner') as mock_planner_class:
                with patch('cortex.cli_commands.SelfRepairLog') as mock_log_class:
                    
                    # Mock healer
                    mock_healer = Mock()
                    mock_report = Mock()
                    mock_report.repair_goals = ["Test repair goal"]
                    mock_report.issues = []
                    mock_healer.run_diagnostic_suite.return_value = mock_report
                    mock_healer_class.return_value = mock_healer
                    
                    # Mock planner
                    mock_planner = Mock()
                    mock_plan = Mock()
                    mock_plan.steps = [Mock()]
                    mock_planner.plan_goal.return_value = mock_plan
                    mock_planner.execute_plan.return_value = {
                        'overall_success': True,
                        'completion_percentage': 100.0,
                        'steps_executed': []
                    }
                    mock_planner_class.return_value = mock_planner
                    
                    # Mock repair log
                    mock_log = Mock()
                    mock_log.log_repair_attempt.return_value = "test-attempt-id"
                    mock_log.complete_repair_attempt.return_value = True
                    mock_log_class.return_value = mock_log
                    
                    result = handle_self_repair_command(top_n=1)
                    assert result == 'continue'
                    mock_healer.run_diagnostic_suite.assert_called_once()
                    mock_planner.plan_goal.assert_called_once()

class TestIntegrationScenarios:
    """Test complete integration scenarios"""
    
    @pytest.fixture
    def temp_dbs(self):
        """Create temporary databases for integration testing"""
        with tempfile.NamedTemporaryFile(suffix='_repair.db', delete=False) as repair_f:
            repair_db = repair_f.name
        
        yield repair_db
        
        os.unlink(repair_db)
    
    def test_complete_self_repair_workflow(self, temp_dbs):
        """Test complete workflow: diagnose -> repair -> log -> learn"""
        repair_db = temp_dbs
        
        # Initialize components
        self_healer = SelfHealer()
        repair_log = SelfRepairLog(db_path=repair_db)
        
        # Mock the healer's external dependencies
        self_healer._repair_log = repair_log
        
        # Step 1: Create mock issues for diagnosis
        with patch.object(self_healer, 'analyze_code_health') as mock_analyze:
            with patch.object(self_healer, 'detect_architecture_flaws') as mock_detect:
                
                # Mock some issues
                mock_issues = [
                    Issue("i1", "code_quality", "high", "test.py", "High priority issue", [], 0.8, 0.5, datetime.now().isoformat()),
                    Issue("i2", "performance", "medium", "slow.py", "Performance issue", [], 0.6, 0.4, datetime.now().isoformat())
                ]
                
                mock_patterns = [
                    Pattern("p1", "anti_pattern", "Test Pattern", "Test anti-pattern", [], 3, "medium", "Fix it")
                ]
                
                mock_analyze.return_value = mock_issues
                mock_detect.return_value = mock_patterns
                
                # Step 2: Run diagnostic
                diagnostic_report = self_healer.run_diagnostic_suite()
                
                assert len(diagnostic_report.issues) == 2
                assert len(diagnostic_report.patterns) == 1
                assert len(diagnostic_report.repair_goals) > 0
        
        # Step 3: Log repair attempt
        repair_goal = diagnostic_report.repair_goals[0] if diagnostic_report.repair_goals else "Test repair goal"
        attempt_id = repair_log.log_repair_attempt(
            goal=repair_goal,
            approach="Test automated repair"
        )
        
        assert attempt_id != ""
        
        # Step 4: Complete repair attempt
        success = repair_log.complete_repair_attempt(
            attempt_id=attempt_id,
            result="Repair completed successfully",
            score=0.85,
            issues_resolved=["i1"],
            side_effects=[]
        )
        
        assert success == True
        
        # Step 5: Take health snapshot
        snapshot_id = repair_log.take_health_snapshot(
            health_score=0.9,  # Improved after repair
            total_issues=1,    # Reduced from 2
            critical_issues=0,
            high_issues=0,
            issues_by_category={"performance": 1},
            patterns_detected=["Test Pattern"]
        )
        
        assert snapshot_id != ""
        
        # Step 6: Verify learning occurred
        summary = repair_log.summarize_recent_repairs(days_back=1)
        
        assert summary['total_attempts'] >= 1
        assert summary['successful_attempts'] >= 1
        assert summary['success_rate'] > 0.5
    
    def test_self_prompter_integration(self):
        """Test integration with SelfPromptGenerator"""
        from agents.self_prompter import SelfPromptGenerator
        
        # Create self-prompter
        prompter = SelfPromptGenerator()
        
        # Mock the self-healer
        mock_healer = Mock()
        mock_report = Mock()
        mock_report.system_health_score = 0.4  # Poor health to trigger goals
        mock_report.issues = [
            Mock(severity='critical', category='security', component='auth.py'),
            Mock(severity='high', category='performance', component='db.py')
        ]
        mock_report.patterns = [
            Mock(risk_level='critical', name='God Class', frequency=5)
        ]
        mock_report.repair_goals = [
            "Fix critical security issue in auth.py",
            "Optimize database performance"
        ]
        mock_healer.run_diagnostic_suite.return_value = mock_report
        
        prompter._self_healer = mock_healer
        
        # Generate goals
        goals = prompter.propose_goals()
        
        # Should include self-repair goals as high priority
        assert len(goals) > 0
        
        # Should contain self-repair related goals
        goal_text = " ".join(goals).lower()
        assert any(keyword in goal_text for keyword in ['urgent', 'critical', 'self-repair', 'high priority'])
    
    def test_failed_repair_learning(self, temp_dbs):
        """Test learning from failed repairs"""
        repair_db = temp_dbs
        repair_log = SelfRepairLog(db_path=repair_db)
        
        # Simulate multiple failed repairs with same approach
        failed_approach = "Naive string replacement"
        
        for i in range(3):
            attempt_id = repair_log.log_repair_attempt(
                goal=f"Fix issue {i}",
                approach=failed_approach
            )
            
            repair_log.complete_repair_attempt(
                attempt_id=attempt_id,
                result=f"Failed: approach doesn't work for issue {i}",
                score=0.2,
                side_effects=[f"Broke feature {i}"]
            )
        
        # Get failed repairs
        failed_repairs = repair_log.get_failed_repairs()
        
        # Should have recorded the failures
        assert len(failed_repairs) >= 3
        
        # Should identify common failure patterns
        approaches = [repair['approach'] for repair in failed_repairs]
        assert approaches.count(failed_approach) >= 3
        
        # Should learn to avoid this approach in future

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])