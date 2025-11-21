// Simple product loader
let products = [];
let currentPage = 1;
let productsPerPage = 6;
let currentSort = 'name-asc';

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function () {
    loadProducts();
});

// Load products from CSV
async function loadProducts() {
    try {
        const response = await fetch('products.csv');
        const csvText = await response.text();
        products = parseCSV(csvText);
        renderPage();
    } catch (error) {
        console.error('Error loading products:', error);
        document.getElementById('products-grid').innerHTML = '<div class="loading">Error loading products</div>';
    }
}

// Simple CSV parser
function parseCSV(csvText) {
    const lines = csvText.trim().split('\n');
    const headers = lines[0].split(',');

    return lines.slice(1).map(line => {
        const values = line.split(',');
        const product = {};

        headers.forEach((header, index) => {
            const value = values[index]?.trim();
            switch (header) {
                case 'id': product[header] = parseInt(value); break;
                case 'price': product[header] = parseFloat(value); break;
                case 'images': product[header] = value.split(',').map(img => img.trim()); break;
                default: product[header] = value;
            }
        });

        return product;
    }).filter(product => product.id);
}

// Render the appropriate page
function renderPage() {
    if (document.getElementById('products-grid')) {
        renderHomePage();
    } else if (document.getElementById('product-detail')) {
        renderProductPage();
    }
}

// Home page rendering
function renderHomePage() {
    renderHeroSlider();
    renderProducts();
    renderPagination();
    updateProductsCount();
}

// Hero slider
function renderHeroSlider() {
    const slider = document.getElementById('hero-slider');
    const featuredProducts = products.slice(0, 3);

    slider.innerHTML = featuredProducts.map((product, index) => `
        <a href="product.html?id=${product.id}" class="slide ${index === 0 ? 'active' : ''}">
            <div class="slide-content">
                <h2>${product.name}</h2>
                <p>${product.short_desc}</p>
                <div class="price">$${product.price}</div>
            </div>
        </a>
    `).join('') + `
        <button class="slider-nav prev-slide" onclick="navigateSlider(-1)">‹</button>
        <button class="slider-nav next-slide" onclick="navigateSlider(1)">›</button>
    `;
}

// Products grid
function renderProducts() {
    const grid = document.getElementById('products-grid');
    const sortedProducts = getSortedProducts();
    const startIndex = (currentPage - 1) * productsPerPage;
    const productsToShow = sortedProducts.slice(startIndex, startIndex + productsPerPage);

    grid.innerHTML = productsToShow.map(product => `
        <a href="product.html?id=${product.id}" class="product-card">
            <div class="product-image">${getInitials(product.name)}</div>
            <h3>${product.name}</h3>
            <p class="short-desc">${product.short_desc}</p>
            <div class="price">$${product.price.toFixed(2)}</div>
        </a>
    `).join('');
}

// Pagination
function renderPagination() {
    const pagination = document.getElementById('pagination');
    const totalPages = Math.ceil(products.length / productsPerPage);

    pagination.innerHTML = `
        <button onclick="changePage(${currentPage - 1})" ${currentPage === 1 ? 'disabled' : ''}>
            Previous
        </button>
        <span class="page-info">Page ${currentPage} of ${totalPages}</span>
        <button onclick="changePage(${currentPage + 1})" ${currentPage === totalPages ? 'disabled' : ''}>
            Next
        </button>
    `;
}

// Product detail page
function renderProductPage() {
    const urlParams = new URLSearchParams(window.location.search);
    const productId = parseInt(urlParams.get('id'));
    const product = products.find(p => p.id === productId);

    const detail = document.getElementById('product-detail');

    if (!product) {
        detail.innerHTML = '<div class="loading"><h2>Product not found</h2></div>';
        return;
    }

    detail.innerHTML = `
        <div class="product-gallery">
            <div class="thumbnail-container">
                ${product.images.map((img, index) => `
                    <div class="thumbnail ${index === 0 ? 'active' : ''}" onclick="changeImage(${index})">
                        ${img}
                    </div>
                `).join('')}
            </div>
            <div class="main-image" onclick="zoomImage()">
                ${getInitials(product.name)}
            </div>
        </div>
        <div class="product-info">
            <h1>${product.name}</h1>
            <div class="product-price">$${product.price.toFixed(2)}</div>
            <div class="long-desc">${product.long_desc}</div>
            <a href="index.html" class="cta-button" style="display: inline-block; margin-top: 2rem;">Back to Products</a>
        </div>
    `;
}

// Helper functions
function getSortedProducts() {
    const [field, order] = currentSort.split('-');

    return [...products].sort((a, b) => {
        let aVal = a[field], bVal = b[field];
        if (field === 'name') {
            aVal = aVal.toLowerCase();
            bVal = bVal.toLowerCase();
        }
        return order === 'asc' ? (aVal < bVal ? -1 : 1) : (aVal > bVal ? -1 : 1);
    });
}

function getInitials(name) {
    return name.split(' ').map(word => word[0]).join('').toUpperCase();
}

function updateProductsCount() {
    const start = (currentPage - 1) * productsPerPage + 1;
    const end = Math.min(currentPage * productsPerPage, products.length);
    document.getElementById('products-count').textContent = `Showing ${start}-${end} of ${products.length} products`;
}

// Event handlers
function navigateSlider(direction) {
    const slides = document.querySelectorAll('.slide');
    let currentIndex = Array.from(slides).findIndex(slide => slide.classList.contains('active'));

    slides[currentIndex].classList.remove('active');
    currentIndex = (currentIndex + direction + slides.length) % slides.length;
    slides[currentIndex].classList.add('active');
}

function sortProducts(sortValue) {
    currentSort = sortValue;
    currentPage = 1;
    renderProducts();
    renderPagination();
    updateProductsCount();
}

function changeProductsPerPage(value) {
    productsPerPage = parseInt(value);
    currentPage = 1;
    renderProducts();
    renderPagination();
    updateProductsCount();
}

function changePage(newPage) {
    const totalPages = Math.ceil(products.length / productsPerPage);
    if (newPage >= 1 && newPage <= totalPages) {
        currentPage = newPage;
        renderProducts();
        renderPagination();
        updateProductsCount();

        // Scroll to products section
        document.getElementById('products').scrollIntoView({ behavior: 'smooth' });
    }
}

function changeImage(index) {
    const thumbnails = document.querySelectorAll('.thumbnail');
    const mainImage = document.querySelector('.main-image');

    thumbnails.forEach(thumb => thumb.classList.remove('active'));
    thumbnails[index].classList.add('active');
    mainImage.textContent = thumbnails[index].textContent;
}

function zoomImage() {
    const mainImage = document.querySelector('.main-image');
    alert(`Zoomed view of: ${mainImage.textContent}`);
}