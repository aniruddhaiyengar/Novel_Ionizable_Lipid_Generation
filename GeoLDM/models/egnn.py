class EGNN_dynamics_QM9(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, attention=False, normalize=False, tanh=False, coords_range=15.0, norm_constant=1, inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum'):
        super(EGNN_dynamics_QM9, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range/norm_constant)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        # Initialize layers with proper initialization
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.zeros_(self.embedding.bias)
        
        self.embedding_out = nn.Linear(self.hidden_nf, in_node_nf)
        nn.init.xavier_uniform_(self.embedding_out.weight)
        nn.init.zeros_(self.embedding_out.bias)
        
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, device=device, act_fn=act_fn, attention=attention, normalize=normalize, tanh=tanh, coords_range=self.coords_range_layer, norm_constant=norm_constant, inv_sublayers=inv_sublayers, sin_embedding=sin_embedding, normalization_factor=self.normalization_factor, aggregation_method=self.aggregation_method))

        self.to(self.device)

    def _forward(self, h, x, edges, edge_attr, node_mask, edge_mask):
        # Input validation and masking
        if torch.isnan(h).any() or torch.isinf(h).any():
            logging.warning("NaN/Inf detected in input h, resetting to zero")
            h = torch.zeros_like(h)
            
        if torch.isnan(x).any() or torch.isinf(x).any():
            logging.warning("NaN/Inf detected in input x, resetting to zero")
            x = torch.zeros_like(x)
            
        # Ensure proper masking with float tensors
        if node_mask is not None:
            node_mask = node_mask.float()  # Convert to float
            if node_mask.dim() == 2:
                node_mask = node_mask.unsqueeze(-1)
            node_mask = (node_mask > 0.5).float()  # Convert to binary float
            h = h * node_mask
            x = x * node_mask
            
        if edge_mask is not None:
            edge_mask = edge_mask.float()  # Convert to float
            if edge_mask.dim() == 2:
                edge_mask = edge_mask.unsqueeze(0)
            edge_mask = (edge_mask > 0.5).float()  # Convert to binary float
            edge_attr = edge_attr * edge_mask
            
        # Initial embedding with validation
        h = self.embedding(h)
        if torch.isnan(h).any():
            logging.warning("NaN detected after initial embedding, resetting to zero")
            h = torch.zeros_like(h)
            
        # Message passing layers with validation
        for i in range(0, self.n_layers):
            h, x = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
            
            # Validate outputs after each layer
            if torch.isnan(h).any() or torch.isinf(h).any():
                logging.warning(f"NaN/Inf detected in h after layer {i}, resetting to zero")
                h = torch.zeros_like(h)
                
            if torch.isnan(x).any() or torch.isinf(x).any():
                logging.warning(f"NaN/Inf detected in x after layer {i}, resetting to zero")
                x = torch.zeros_like(x)
                
            # Re-apply masking after each layer
            if node_mask is not None:
                h = h * node_mask
                x = x * node_mask
                
        # Final embedding with validation
        h = self.embedding_out(h)
        if torch.isnan(h).any():
            logging.warning("NaN detected after final embedding, resetting to zero")
            h = torch.zeros_like(h)
            
        # Apply final masking
        if node_mask is not None:
            h = h * node_mask
            x = x * node_mask
            
        return h, x

    def forward(self, h, x, node_mask=None, edge_mask=None):
        # Input validation
        if torch.isnan(h).any() or torch.isinf(h).any():
            logging.warning("NaN/Inf detected in input h, resetting to zero")
            h = torch.zeros_like(h)
            
        if torch.isnan(x).any() or torch.isinf(x).any():
            logging.warning("NaN/Inf detected in input x, resetting to zero")
            x = torch.zeros_like(x)
            
        # Ensure proper masking
        if node_mask is not None:
            node_mask = node_mask.float()  # Convert to float
            if node_mask.dim() == 2:
                node_mask = node_mask.unsqueeze(-1)
            node_mask = (node_mask > 0.5).float()  # Convert to binary float
            h = h * node_mask
            x = x * node_mask
            
        # Create edges and edge attributes
        edges = self._get_edges(x)
        edge_attr = self._get_edge_attr(edges, x)
        
        # Apply edge masking if provided
        if edge_mask is not None:
            edge_mask = edge_mask.float()  # Convert to float
            if edge_mask.dim() == 2:
                edge_mask = edge_mask.unsqueeze(0)
            edge_mask = (edge_mask > 0.5).float()  # Convert to binary float
            edge_attr = edge_attr * edge_mask
            
        # Forward pass with validation
        h, x = self._forward(h, x, edges, edge_attr, node_mask, edge_mask)
        
        # Final validation
        if torch.isnan(h).any() or torch.isinf(h).any():
            logging.warning("NaN/Inf detected in final h, resetting to zero")
            h = torch.zeros_like(h)
            
        if torch.isnan(x).any() or torch.isinf(x).any():
            logging.warning("NaN/Inf detected in final x, resetting to zero")
            x = torch.zeros_like(x)
            
        # Apply final masking
        if node_mask is not None:
            h = h * node_mask
            x = x * node_mask
            
        return h, x

    def _get_edges(self, x):
        # Create edges with validation
        rows, cols = [], []
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                if i != j:
                    rows.append(i)
                    cols.append(j)
        edges = [torch.tensor(rows, device=self.device), torch.tensor(cols, device=self.device)]
        return edges

    def _get_edge_attr(self, edges, x):
        # Calculate edge attributes with validation
        row, col = edges
        
        # Calculate squared differences with overflow protection
        diff = x[row] - x[col]
        squared_diff = torch.clamp(diff ** 2, max=1e6)  # Prevent overflow
        
        # Sum with overflow protection
        edge_attr = torch.clamp(torch.sum(squared_diff, 1), max=1e6)
        
        # Safe square root with epsilon
        edge_attr = torch.sqrt(edge_attr + 1e-6)
        
        # Validate edge attributes
        if torch.isnan(edge_attr).any() or torch.isinf(edge_attr).any():
            logging.warning("NaN/Inf detected in edge attributes, resetting to zero")
            edge_attr = torch.zeros_like(edge_attr)
            
        return edge_attr 