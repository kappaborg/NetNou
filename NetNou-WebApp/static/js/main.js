/**
 * NetNou - Main JavaScript File
 * Contains shared functionality across the application
 */

// Utility functions
const NetNou = {
    // Flash message handling
    initFlashMessages: function() {
        const flashMessages = document.querySelectorAll('.flash-message');
        
        flashMessages.forEach(message => {
            // Auto-dismiss flash messages after 5 seconds
            setTimeout(() => {
                message.classList.add('fade-out');
                setTimeout(() => {
                    message.remove();
                }, 500); // Wait for fade animation
            }, 5000);
            
            // Add close button
            const closeBtn = document.createElement('span');
            closeBtn.classList.add('close-btn');
            closeBtn.innerHTML = '&times;';
            closeBtn.addEventListener('click', () => {
                message.remove();
            });
            
            message.appendChild(closeBtn);
        });
    },
    
    // Lazy loading images
    lazyLoadImages: function() {
        if ('loading' in HTMLImageElement.prototype) {
            // Browser supports native lazy loading
            const lazyImages = document.querySelectorAll('img[data-src]');
            lazyImages.forEach(img => {
                img.src = img.dataset.src;
                img.loading = 'lazy';
            });
        } else {
            // Fallback for browsers that don't support native lazy loading
            const lazyImages = document.querySelectorAll('img[data-src]');
            
            if ('IntersectionObserver' in window) {
                const imageObserver = new IntersectionObserver((entries) => {
                    entries.forEach(entry => {
                        if (entry.isIntersecting) {
                            const img = entry.target;
                            img.src = img.dataset.src;
                            imageObserver.unobserve(img);
                        }
                    });
                });
                
                lazyImages.forEach(img => {
                    imageObserver.observe(img);
                });
            } else {
                // Fallback for older browsers without IntersectionObserver
                let active = false;
                
                const lazyLoad = function() {
                    if (active === false) {
                        active = true;
                        
                        setTimeout(() => {
                            lazyImages.forEach(img => {
                                if ((img.getBoundingClientRect().top <= window.innerHeight && img.getBoundingClientRect().bottom >= 0) && getComputedStyle(img).display !== 'none') {
                                    img.src = img.dataset.src;
                                    img.removeAttribute('data-src');
                                    
                                    if (lazyImages.length === 0) {
                                        document.removeEventListener('scroll', lazyLoad);
                                        window.removeEventListener('resize', lazyLoad);
                                        window.removeEventListener('orientationChange', lazyLoad);
                                    }
                                }
                            });
                            
                            active = false;
                        }, 200);
                    }
                };
                
                document.addEventListener('scroll', lazyLoad);
                window.addEventListener('resize', lazyLoad);
                window.addEventListener('orientationChange', lazyLoad);
            }
        }
    },
    
    // Debounce function for performance optimization
    debounce: function(func, wait = 300) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },
    
    // Initialize all common functionality
    init: function() {
        // Initialize flash messages
        this.initFlashMessages();
        
        // Initialize lazy loading
        this.lazyLoadImages();
        
        // Add page-loaded class for transitions
        document.body.classList.add('page-loaded');
    }
};

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    NetNou.init();
}); 