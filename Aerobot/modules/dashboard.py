import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import modules.data_utils as data_utils
import os
import base64

@st.cache_data
def get_base64_image(image_path):
    """Converts a local image to base64 so HTML can display it."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        return None

def style_chart(fig):
    is_dark = st.session_state.get('is_dark', False)
    if is_dark:
        bg_color = '#1F2937'
        text_color = '#E0E0E0'
        grid_color = '#374151'
    else:
        bg_color = '#FFFFFF'
        text_color = '#2D2D2D'
        grid_color = '#F0F2F6'

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=grid_color)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=grid_color)
    return fig

def show_dashboard():
    col_icon, col_title = st.columns([0.08, 0.92])
    with col_icon:
        st.markdown("""
        <div style="padding-top: 10px;">
            <svg width="45" height="45" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M21 21H3V3" stroke="#00BFFF" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M21 9L13.5 16.5L9 12L3 18" stroke="#00BFFF" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M21 9V13" stroke="#00BFFF" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M21 9H17" stroke="#00BFFF" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        </div>
        """, unsafe_allow_html=True)
        
    with col_title:
        st.title("Operational Dashboard")
    
    st.markdown("---")

    df = data_utils.load_data()

    col_upload, col_image = st.columns([1, 3])

    with col_upload:
        st.markdown("##### 📤 Upload Data")
        with st.container(border=True):
            uploaded_file = st.file_uploader("Upload CSV", type=['csv'], label_visibility="collapsed")
            if uploaded_file:
                try:
                    new_df = pd.read_csv(uploaded_file)
                    df = data_utils.engineer_financial_features(new_df)
                    st.success("Loaded!")
                except Exception as e:
                    st.error(f"Error: {e}")

    with col_image:
        # --- SMART IMAGE SEARCH (Fixes Case Sensitivity Issues) ---
        # List all possible variations to try
        possible_paths = [
            "assets/dashboard_cover3.png",
            "assets/dashboard_cover3.PNG",
            "assets/Dashboard_Cover3.png",
            "assets/dashboard_cover.png",
            "assets/dashboard_cover.jpg"
        ]
        
        found_path = None
        for p in possible_paths:
            if os.path.exists(p):
                found_path = p
                break
        
        img_src = ""
        if found_path:
            base64_img = get_base64_image(found_path)
            if base64_img:
                # Basic mime type detection
                mime = "image/png" if found_path.lower().endswith(".png") else "image/jpeg"
                img_src = f"data:{mime};base64,{base64_img}"
        
        # Fallback if NOTHING is found
        if not img_src:
            img_src = "https://images.unsplash.com/photo-1436491865332-7a61a109cc05?q=80&w=2074&auto=format&fit=crop"
            
            # --- DEBUGGER (Visible only if image fails) ---
            # This helps you see exactly what files made it to the cloud
            if os.path.exists("assets"):
                files = os.listdir("assets")
                st.caption(f"⚠️ Debug: Could not find 'dashboard_cover3.png'. Files in 'assets' folder: {files}")
            else:
                st.caption("⚠️ Debug: 'assets' folder is missing on the server.")

        st.markdown(f"""
            <div style="
                width: 100%; 
                height: 250px; 
                border-radius: 12px; 
                overflow: hidden; 
                border: 1px solid #374151;
                display: flex;
                align-items: center;
                justify-content: center;
            ">
                <img src="{img_src}" style="
                    width: 100%; 
                    height: 100%; 
                    object-fit: cover; 
                    object-position: center;
                ">
            </div>
        """, unsafe_allow_html=True)

    if df is None:
        st.warning("No data available.")
        return

    st.markdown("<br>", unsafe_allow_html=True)

    total_revenue = df['Revenue'].sum()
    total_passengers = len(df)
    avg_delay = df.get('Departure Delay in Minutes', pd.Series([0])).mean()
    profit_margin = (df['Profit'].sum() / total_revenue * 100) if total_revenue > 0 else 0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Revenue", f"${total_revenue/1_000_000:.1f}M", "+12.5%")
    kpi2.metric("Total Flights", f"{total_passengers:,}", "+2.1%")
    kpi3.metric("Avg Delay", f"{avg_delay:.1f} min", "-3.3%")
    kpi4.metric("Profit Margin", f"{profit_margin:.1f}%", "+1.2%")

    st.markdown("---")

    col_trend, col_pie = st.columns([3, 1])
    with col_trend:
        st.subheader("📈 Revenue Trends")
        if 'Date' in df.columns:
            trend_df = df.groupby(df['Date'].dt.to_period("M")).agg({'Revenue': 'sum'}).reset_index()
            trend_df['Date'] = trend_df['Date'].dt.to_timestamp()
            fig_trend = px.line(trend_df, x='Date', y='Revenue', markers=True)
            fig_trend.update_traces(line_color='#3B82F6', line_width=3)
            fig_trend.update_layout(height=350)
            fig_trend = style_chart(fig_trend)
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Date column missing.")

    with col_pie:
        st.subheader("Revenue by Class")
        if 'Class' in df.columns:
            fig_pie = px.pie(df, names='Class', values='Revenue', hole=0.6, color_discrete_sequence=px.colors.sequential.RdBu)
            fig_pie.update_layout(height=350, showlegend=False)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie = style_chart(fig_pie)
            st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("⛽ Cost Efficiency")
    if 'Fuel_Cost' in df.columns:
        is_dark = st.session_state.get('is_dark', False)
        border_color = '#0E1117' if is_dark else '#FFFFFF'
        fig_scatter = px.scatter(df.head(300), x='Fuel_Cost', y='Profit', color='Class', 
                                 size='Flight Distance', size_max=12, opacity=0.9)
        fig_scatter.update_traces(marker=dict(line=dict(width=1, color=border_color)))
        fig_scatter.update_layout(height=400)
        fig_scatter = style_chart(fig_scatter)
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    st.subheader("✈️ Network Performance")
    col_dest, col_routes = st.columns([1, 3])
    with col_dest:
        st.caption("Top 5 Destinations")
        city_df = df['Destination'].value_counts().nlargest(5).reset_index()
        city_df.columns = ['City', 'Passengers']
        fig_city = px.bar(city_df, x='City', y='Passengers', color='Passengers', 
                          color_continuous_scale='Blues')
        fig_city.update_layout(height=350, showlegend=False)
        fig_city = style_chart(fig_city)
        st.plotly_chart(fig_city, use_container_width=True)

    with col_routes:
        st.caption("Top 5 Performing Routes")
        route_df = df.groupby('Route')['Revenue'].sum().nlargest(5).reset_index()
        fig_routes = px.bar(route_df, x='Revenue', y='Route', orientation='h', 
                            text_auto='.2s', color='Revenue', color_continuous_scale='Viridis')
        fig_routes.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, height=350)
        fig_routes = style_chart(fig_routes)
        st.plotly_chart(fig_routes, use_container_width=True)

    st.markdown("---")

    st.subheader("🗺️ Live Flight Routes")
    if 'Origin_Lat' in df.columns:
        map_df = df.groupby(['Route', 'Origin', 'Destination', 'Origin_Lat', 'Origin_Lon', 'Dest_Lat', 'Dest_Lon']) \
                   .agg({'Revenue': 'sum', 'Profit': 'count'}).reset_index()
        map_df = map_df.sort_values('Revenue', ascending=False).head(20)
        fig_map = go.Figure()
        for i, row in map_df.iterrows():
            fig_map.add_trace(go.Scattergeo(
                lon = [row['Origin_Lon'], row['Dest_Lon']],
                lat = [row['Origin_Lat'], row['Dest_Lat']],
                mode = 'lines',
                line = dict(width=2, color='#00BFFF'), 
                opacity = 0.6,
                name = row['Route'],
                hoverinfo = 'text',
                text = f"{row['Route']}: ${row['Revenue']/1000:.0f}k"
            ))
        origins = map_df[['Origin', 'Origin_Lat', 'Origin_Lon']].rename(columns={'Origin':'Code', 'Origin_Lat':'Lat', 'Origin_Lon':'Lon'})
        dests = map_df[['Destination', 'Dest_Lat', 'Dest_Lon']].rename(columns={'Destination':'Code', 'Dest_Lat':'Lat', 'Dest_Lon':'Lon'})
        airports = pd.concat([origins, dests]).drop_duplicates(subset='Code')
        fig_map.add_trace(go.Scattergeo(
            lon = airports['Lon'],
            lat = airports['Lat'],
            hoverinfo = 'text',
            text = airports['Code'],
            mode = 'markers',
            marker = dict(size=6, color='#10B981', line=dict(width=1, color='white'))
        ))
        is_dark = st.session_state.get('is_dark', False)
        if is_dark:
            land_c = '#374151'
            ocean_c = '#1F2937'
            bg_c = '#1F2937'
            country_c = '#0E1117'
            text_c = '#E0E0E0'
        else:
            land_c = '#F0F2F6'
            ocean_c = '#FFFFFF'
            bg_c = '#FFFFFF'
            country_c = '#D1D5DB'
            text_c = '#2D2D2D'
        fig_map.update_layout(
            showlegend = False,
            geo = dict(
                projection_type = 'equirectangular',
                showland = True,
                landcolor = land_c,
                oceancolor = ocean_c,
                showocean = True,
                countrycolor = country_c,
                coastlinecolor = country_c,
                bgcolor = bg_c
            ),
            height = 600,
            margin = dict(l=0, r=0, t=0, b=0),
            paper_bgcolor=bg_c,
            font_color=text_c
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("Geographic data not available.")

    with st.expander("📂 View Detailed Data"):
        st.dataframe(df)