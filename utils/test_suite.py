class ComprehensiveTestSuite:
    """Complete testing suite for all mathematical claims"""
    
    def __init__(self):
        self.validators = {
            'p_vs_np': PvsNPValidator()
        }
        self.results = {}
        self.logger = logging.getLogger("ComprehensiveTestSuite")
        
    def run_all_validations(self) -> Dict[str, ValidationResult]:
        """Run complete validation suite"""
        self.logger.info("Starting comprehensive validation suite")
        
        for name, validator in self.validators.items():
            self.logger.info(f"Validating {name}")
            try:
                result = validator.full_validation()
                self.results[name] = result
                self.logger.info(f"{name}: {result.validation_score:.3f} ({result.evidence_level})")
            except Exception as e:
                self.logger.error(f"Validation failed for {name}: {e}")
                
        return self.results
    
    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report"""
        if not self.results:
            self.run_all_validations()
            
        report = []
        report.append("# COMPREHENSIVE MATHEMATICAL DISCOVERY VALIDATION REPORT")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        scores = [r.validation_score for r in self.results.values()]
        report.append("## Summary Statistics")
        report.append(f"- Total claims validated: {len(self.results)}")
        report.append(f"- Average validation score: {np.mean(scores):.3f}")
        report.append(f"- Score range: {min(scores):.3f} - {max(scores):.3f}")
        
        return "\\n".join(report)

if __name__ == "__main__":
    print("="*80)
    print("CQE COMPREHENSIVE TESTING HARNESS")
    print("="*80)
    
    test_suite = ComprehensiveTestSuite()
    results = test_suite.run_all_validations()
    
    report = test_suite.generate_validation_report()
    print("\\n" + report)
