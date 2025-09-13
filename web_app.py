#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask web application for similarity scoring

Usage:
    python3 web_app.py

Then open http://localhost:5000 in your browser
"""

from flask import Flask, render_template, request, jsonify
from similarity_scorer import SimilarityScorer
import re

app = Flask(__name__)
scorer = SimilarityScorer()

@app.route('/')
def index():
    """Main page with input form"""
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate_similarity():
    """Calculate similarity scores for input strings"""
    try:
        # Get input text and split into strings
        input_text = request.json.get('strings', '')
        
        # Get weights from request (default to original weights if not provided)
        jaccard_weight = float(request.json.get('jaccard_weight', 0.0))
        jaro_winkler_weight = float(request.json.get('jaro_winkler_weight', 0.9))
        acronym_weight = float(request.json.get('acronym_weight', 0.1))
        
        # Split by newlines and filter out empty strings
        strings = [s.strip() for s in input_text.split('\n') if s.strip()]
        
        if len(strings) < 2:
            return jsonify({
                'error': '请提供至少 2 个字符串（每行一个）'
            }), 400
        
        results = []
        
        if len(strings) == 2:
            # Two strings: detailed comparison
            s1, s2 = strings
            result = scorer.calculate_similarity(s1, s2)
            
            # Calculate custom overall score with user weights
            custom_overall = (jaccard_weight * result.jaccard_score + 
                            jaro_winkler_weight * result.jaro_winkler_score + 
                            acronym_weight * result.acronym_score)
            
            results.append({
                'type': 'detailed',
                'string1': s1,
                'string2': s2,
                'normalized1': scorer.normalize_text(s1),
                'normalized2': scorer.normalize_text(s2),
                'jaccard': round(result.jaccard_score, 3),
                'jaro_winkler': round(result.jaro_winkler_score, 3),
                'acronym': round(result.acronym_score, 3),
                'overall': round(custom_overall, 3),
                'weights': {
                    'jaccard': jaccard_weight,
                    'jaro_winkler': jaro_winkler_weight,
                    'acronym': acronym_weight
                }
            })
            
        else:
            # Multiple strings: all pairs comparison
            for i in range(len(strings)):
                for j in range(i + 1, len(strings)):
                    s1, s2 = strings[i], strings[j]
                    result = scorer.calculate_similarity(s1, s2)
                    
                    # Calculate custom overall score with user weights
                    custom_overall = (jaccard_weight * result.jaccard_score + 
                                    jaro_winkler_weight * result.jaro_winkler_score + 
                                    acronym_weight * result.acronym_score)
                    
                    results.append({
                        'type': 'pair',
                        'string1': s1,
                        'string2': s2,
                        'jaccard': round(result.jaccard_score, 3),
                        'jaro_winkler': round(result.jaro_winkler_score, 3),
                        'acronym': round(result.acronym_score, 3),
                        'overall': round(custom_overall, 3)
                    })
        
        return jsonify({
            'success': True,
            'results': results,
            'total_strings': len(strings),
            'total_pairs': len(results)
        })
        
    except Exception as e:
        return jsonify({
            'error': f'计算相似度时出错：{str(e)}'
        }), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5001))
    print("正在启动相似度计算器网页应用...")
    print(f"请在浏览器中打开 http://localhost:{port}")
    app.run(debug=False, host='0.0.0.0', port=port)
